"""OpenAI-compatible annotation tool-agent runner."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from app.annotation_providers.registry import ProviderConfigurationError
from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_openai_extra_tool_schemas,
    annotation_openai_tool_schemas,
    build_tool_agent_system_prompt,
    tool_agent_response,
    tool_agent_stage,
)
from app.shared.contracts import AgentRequest, AgentResponse

_NODE = "openai_tool_agent"

def _client(request: AgentRequest):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderConfigurationError(
            "The OpenAI annotation provider requires the `openai` package."
        ) from exc

    opts = request.config.provider_options
    api_key_env = str(opts.get("api_key_env") or "OPENAI_API_KEY")
    api_key = str(opts.get("api_key") or os.getenv(api_key_env) or "")
    if not api_key:
        raise ProviderConfigurationError(
            f"Annotation provider {request.provider_id!r} requires API key env var {api_key_env}."
        )

    base_url = str(opts.get("base_url") or "").strip() or None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _call_tool(surface: AnnotationToolSurface, name: str, args: Dict[str, Any]) -> Tuple[str, List[str]]:
    _result, text, images = surface.call_tool(name, args)
    return text, images


def run_openai_compatible(request: AgentRequest) -> AgentResponse:
    model = request.config.model.strip()
    if not model:
        raise ProviderConfigurationError(
            f"Annotation provider {request.provider_id!r} requires a model."
        )

    client = _client(request)
    capture = ToolAgentCapture(
        node_name=_NODE,
        cur_start=int(request.parent_start),
        cur_end=int(request.parent_end),
    )
    surface = AnnotationToolSurface(request, capture)
    cb = request.callbacks
    if cb.vlm_prompt:
        cb.vlm_prompt(request.planner_prompt, tool_agent_stage(_NODE, "main"))
    if cb.progress:
        cb.progress(_NODE, "session starting")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_tool_agent_system_prompt(request)},
        {"role": "user", "content": request.planner_prompt},
    ]
    max_turns = int(request.config.provider_options.get("max_turns") or 30)

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                *annotation_openai_tool_schemas(),
                *annotation_openai_extra_tool_schemas(request),
            ],
            tool_choice="auto",
            temperature=float(request.config.temperature),
            max_tokens=int(request.config.max_new_tokens),
        )
        message = response.choices[0].message
        content = message.content or ""
        if content:
            capture.text_chunks.append(content)
            if cb.vlm_stream:
                cb.vlm_stream(content)
        messages.append(message.model_dump(exclude_none=True))
        tool_calls = message.tool_calls or []
        if not tool_calls:
            break
        for call in tool_calls:
            capture.tool_calls += 1
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if cb.progress:
                cb.progress(_NODE, f"tool {capture.tool_calls}: {name}")
            text, images = _call_tool(surface, name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": text,
            })
            for img_b64 in images:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image returned by `{name}`."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                })
        if capture.submitted:
            break

    if cb.progress:
        cb.progress(
            _NODE,
            f"done - {capture.tool_calls} tool call(s), submitted={capture.submitted}, revised={capture.revised}",
        )
    return tool_agent_response(capture, request)
