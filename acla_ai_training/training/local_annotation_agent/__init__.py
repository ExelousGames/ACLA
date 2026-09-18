"""
The agent — the box.

Exposes one public entry point:

    from training.local_annotation_agent import run_agent, AgentRequest, AgentResponse
    response = run_agent(request)

The box is domain-free. It runs provider-specific harnesses and captures
structured submissions. It does NOT know what the caller wants — that
intent rides in ``planner_prompt`` on the AgentRequest.

Sub-modules:
    contracts       Public dataclasses crossing the box boundary.
    evaluators      Format + evidence evaluator suite + formatter registry.
    backends        claude_sdk (Claude Agent SDK), OpenAI-compatible providers.
    tools           Annotation-domain helpers.
    runners         Claude / OpenAI execution paths.
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
    from training.annotation_providers.registry import (
        get_annotation_provider,
        validate_provider_ready,
    )

    provider = get_annotation_provider(request.provider_id)
    validate_provider_ready(provider)

    if provider.runner == "claude_cli":
        from training.annotation_providers.claude_runner import run_claude
        return run_claude(request)
    if provider.runner == "openai_compatible":
        from training.annotation_providers.openai_runner import run_openai_compatible
        return run_openai_compatible(request)
    raise ValueError(
        f"unknown annotation provider runner {provider.runner!r} for {provider.id!r}"
    )


def __getattr__(name: str):
    if name == "ClaudeUsageExhausted":
        from training.annotation_providers.claude_runner import ClaudeUsageExhausted
        return ClaudeUsageExhausted
    raise AttributeError(name)
