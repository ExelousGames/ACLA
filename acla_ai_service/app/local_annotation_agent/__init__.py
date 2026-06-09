"""
The agent — the box.

Exposes one public entry point:

    from app.local_annotation_agent import run_agent, AgentRequest, AgentResponse
    response = run_agent(request)

The box is domain-free. It knows how to plan, render telemetry graphs,
run deterministic queries, and synthesise responses. It does NOT know
what the caller wants — that intent rides in ``planner_prompt`` and
``synth_prompt`` on the AgentRequest. The caller also names the root
Agent to invoke via ``request.extra_state["root_agent"]`` and may
register additional Claude tools via ``extra_state["claude_extra_tools"]``.

Sub-modules:
    contracts       Public dataclasses crossing the box boundary.
    framework       Planner/executor/synthesizer/evaluator topology.
    evaluators      Format + evidence evaluator suite + formatter registry.
    backends        local_vlm (llama-server), claude_sdk (Claude Agent SDK).
    sub_agents      describe_graphs, zoom — generic plan-step capabilities.
    tools           Telemetry graph rendering + query dispatchers.
    runners         local (LangGraph) and claude (agentic) execution paths.
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
from app.claude_annotation_agent.runner import ClaudeUsageExhausted

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

    if provider.runner == "local_vlm":
        from app.local_annotation_agent.runner import run_local
        return run_local(request)
    if provider.runner == "claude_cli":
        from app.claude_annotation_agent.runner import run_claude
        return run_claude(request)
    if provider.runner == "openai_compatible":
        from app.annotation_providers.openai_runner import run_openai_compatible
        return run_openai_compatible(request)
    raise ValueError(
        f"unknown annotation provider runner {provider.runner!r} for {provider.id!r}"
    )
