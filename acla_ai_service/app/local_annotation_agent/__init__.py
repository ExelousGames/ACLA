"""
The agent — the box.

Exposes one public entry point:

    from app.local_annotation_agent import run_agent, AgentRequest, AgentResponse
    response = run_agent(request)

The box is domain-free. It knows how to run deterministic queries and
capture structured submissions. It does NOT know what the caller wants —
that intent rides in ``planner_prompt`` on the AgentRequest. Providers use
the shared tool-agent contract and may register additional tools via
``extra_state["tool_agent_extra_tools"]``.

Sub-modules:
    contracts       Public dataclasses crossing the box boundary.
    framework       Planner/executor/synthesizer/evaluator topology.
    evaluators      Format + evidence evaluator suite + formatter registry.
    backends        claude_sdk (Claude Agent SDK), OpenAI-compatible providers.
    sub_agents      label verification and non-visual plan-step capabilities.
    tools           Telemetry graph rendering + query dispatchers.
    runners         local / Claude / OpenAI execution paths.
"""

from __future__ import annotations

from app.shared.contracts import (
    AgentCallbacks,
    AgentRequest,
    AgentResponse,
    Attachment,
    ProviderConfig,
    StepEvent,
)
from app.annotation_providers.registry import (
    get_annotation_provider,
    validate_provider_ready,
)
from app.annotation_providers.claude_runner import ClaudeUsageExhausted

BackendConfig = ProviderConfig

__all__ = [
    "AgentCallbacks",
    "AgentRequest",
    "AgentResponse",
    "Attachment",
    "BackendConfig",
    "ProviderConfig",
    "ClaudeUsageExhausted",
    "StepEvent",
    "run_agent",
]


def run_agent(request: AgentRequest) -> AgentResponse:
    """Dispatch to the selected annotation provider."""
    provider = get_annotation_provider(request.provider_id)
    validate_provider_ready(provider)

    if provider.runner == "claude_cli":
        from app.annotation_providers.claude_runner import run_claude
        return run_claude(request)
    if provider.runner == "openai_compatible":
        from app.annotation_providers.openai_runner import run_openai_compatible
        return run_openai_compatible(request)
    if provider.runner == "local_pipeline":
        from app.local_annotation_agent.runner import run_local
        return run_local(request)
    raise ValueError(
        f"unknown annotation provider runner {provider.runner!r} for {provider.id!r}"
    )
