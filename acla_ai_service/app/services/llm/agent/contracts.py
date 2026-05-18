"""
Public contracts that cross the agent boundary.

Everything in this file is domain-free. The box accepts an AgentRequest
and returns an AgentResponse. Callers (flows, pipelines, UIs) translate
their domain intent into prompts and parse the raw output back into
domain types — neither lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# Backend selection + config
# ---------------------------------------------------------------------------


Backend = Literal["local", "claude"]


@dataclass
class BackendConfig:
    """Backend-specific knobs. Only the fields matching the active backend
    are read; the rest are ignored.

    ``backend == "local"`` reads gguf_path / mmproj_path / hf_repo /
    quantization_type / context_size / n_gpu_layers.

    ``backend == "claude"`` reads claude_model / claude_use_thinking.

    Generation knobs (max_new_tokens, temperature) apply to both.
    """

    max_new_tokens: int = 1500
    temperature: float = 0.7

    # local VLM (llama-server)
    gguf_path: Optional[str] = None
    mmproj_path: Optional[str] = None
    hf_repo: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization_type: str = "Q4_K_M"
    context_size: int = 32768
    n_gpu_layers: int = -1

    # claude
    claude_model: str = "claude-sonnet-4-6"
    claude_use_thinking: bool = False


# ---------------------------------------------------------------------------
# Callbacks the caller plugs in to observe the run
# ---------------------------------------------------------------------------


class AgentCallbacks(Protocol):
    """Callback surface the runners drive during a run.

    Every member is optional — runners check before calling. The protocol
    exists so callers get IDE completion and so the contract is documented
    in one place instead of as five loose ``Optional[Callable]`` kwargs.
    """

    progress: Optional[Callable[[str, str], None]]
    """``progress(node_name, detail)`` — node-level milestones."""

    vlm_prompt: Optional[Callable[[str, Dict[str, Any]], None]]
    """``vlm_prompt(prompt, stage)`` — full prompt just before VLM call."""

    vlm_stream: Optional[Callable[[str], None]]
    """``vlm_stream(chunk)`` — streamed text tokens."""

    vlm_reasoning: Optional[Callable[[str], None]]
    """``vlm_reasoning(chunk)`` — streamed thinking blocks (claude only)."""

    step_event: Optional[Callable[[str, Dict[str, Any]], None]]
    """``step_event(summary, stage)`` — non-VLM events (renders, queries, tools)."""


@dataclass
class NoopCallbacks:
    """Default ``AgentCallbacks`` implementation with all hooks set to None."""

    progress: Optional[Callable[[str, str], None]] = None
    vlm_prompt: Optional[Callable[[str, Dict[str, Any]], None]] = None
    vlm_stream: Optional[Callable[[str], None]] = None
    vlm_reasoning: Optional[Callable[[str], None]] = None
    step_event: Optional[Callable[[str, Dict[str, Any]], None]] = None


# ---------------------------------------------------------------------------
# Attachments — opaque payloads the caller seeds and the agent passes around
# ---------------------------------------------------------------------------


@dataclass
class Attachment:
    """A piece of context the caller hands the agent or the agent emits.

    The agent does not interpret ``content`` — it forwards attachments
    through the pool, slices them per-step, and renders them into prompts.
    The caller decides what shape ``content`` takes (a dict, a string,
    raw bytes, a DataFrame ref — anything).

    ``content_schema`` is a free-form string the agent uses to dispatch
    rendering (e.g. ``"parent_segment"`` selects one renderer template
    from the framework's library). Treat it as a hint to renderers, not
    a contract enforced by the agent.
    """

    name: str
    kind: Literal["text", "structured", "image", "binary"]
    label: str
    content: Any
    content_schema: Optional[str] = None


@dataclass
class StepEvent:
    """A single event emitted during a run — node start/end, tool call, etc.

    Aggregated into ``AgentResponse.step_events`` so the caller can render
    a transcript without subscribing to live callbacks.
    """

    stage: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request — what the caller hands the box
# ---------------------------------------------------------------------------


@dataclass
class AgentRequest:
    """The full input to a run.

    Mandatory fields express *intent* (prompts) and *context* (df, range,
    initial attachments). The box reads these but never invents domain
    semantics on top of them — the caller's prompts and attachments are
    the only source of intent.

    Fields:
        backend             which execution path to take.
        config              backend-specific knobs.
        planner_prompt      full text the planner sends to the VLM. The
                            planner decides which sub-agents to schedule;
                            constrain that choice in natural language here.
        synth_prompt        callable (state) -> (intro, outro). Always a
                            callable so the caller can read state /
                            attachments populated by earlier steps before
                            building the synth prompt. Static prompts wrap
                            as ``lambda _: (intro, outro)``.
        df_ref              opaque dataset handle the tools dereference.
        parent_start/end    the working iloc range for this run.
        initial_attachments seed pool entries the caller pre-builds.
        callbacks           observability hooks.
        session_id          opaque tag forwarded to step events for audit.
        extra_state         free-form bag forwarded into the runner's
                            initial state. The local runner reads
                            ``extra_state["root_agent"]`` to pick which
                            registered Agent to invoke. Beyond that the
                            framework does not inspect or mutate keys —
                            only those declared in the active Agent's
                            state schema survive LangGraph's filtering.
    """

    backend: Backend
    config: BackendConfig
    planner_prompt: str
    synth_prompt: Callable[[Dict[str, Any]], Tuple[str, str]]
    df_ref: Any
    parent_start: int
    parent_end: int
    initial_attachments: List[Attachment] = field(default_factory=list)
    callbacks: AgentCallbacks = field(default_factory=NoopCallbacks)
    session_id: str = ""
    extra_state: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response — what the box hands back
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """The full output of a run.

    The agent returns raw text. The caller is responsible for parsing it
    into whatever typed result its flow defines. ``attachments`` lets the
    caller surface intermediate artifacts (rendered graph PNGs, planner
    plan text, observation paragraphs, sub-agent emissions) to the UI
    without re-deriving them. Any domain-specific output (verified
    shortlists, label scores, etc.) rides through ``attachments`` —
    there are no domain-named fields on this dataclass.

    Fields:
        raw_response        the synthesizer's final text (post-evaluator).
        verdict             ``"pass"`` / ``"fail"`` / ``""`` — evaluator verdict.
        attachments         everything the run produced, keyed by name.
        step_events         transcript of node/tool events for replay/audit.
        graph_images        convenience accessor for rendered PNGs (bytes).
        plan_steps          the planner's chosen plan (for debugging/UI).
        messages            role/content trace from planner/synth/sub-agents.
    """

    raw_response: str
    verdict: str = ""
    attachments: Dict[str, Attachment] = field(default_factory=dict)
    step_events: List[StepEvent] = field(default_factory=list)
    graph_images: List[bytes] = field(default_factory=list)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
