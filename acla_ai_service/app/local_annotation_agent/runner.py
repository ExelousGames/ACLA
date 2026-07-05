"""
Local runner — drives the shared annotation tool-agent surface via the
local VLM service.

Current provider topology:

    same tool-agent prompt as Claude/OpenAI ──► local llama-server
        └─ tool calls through AnnotationToolSurface ──► submit_result

Legacy LangGraph topology (available through the exported helpers):

    planner ──► executor (loops per plan step) ──► synthesizer ──► evaluator ──► END

The legacy LangGraph helpers below are still exported for root Agent classes,
but the provider entrypoint uses the same prompt + submit_result contract as
the Claude and OpenAI annotation providers.

The legacy graph runner:
  1. Wires the VLM/LLM callables into the shared eval-LLM holder so every
     sub-agent and evaluator picks them up.
  2. Seeds the initial graph state from the AgentRequest (df_ref, range,
     attachments, planner_prompt, synth_prompt, extra_state).
  3. Streams the compiled graph, capturing node events into a transcript.
  4. Returns an AgentResponse — the synthesiser's raw text plus every
     attachment/graph-image/message the run produced.

This runner contains NO domain logic. The provider entrypoint sends
``request.planner_prompt`` verbatim as the user message and uses the same
tool definitions / ``submit_result`` capture as the other annotation
providers.

The phase helpers ``default_planner_node`` / ``default_synth_node`` /
``default_eval_node`` are exported so a root Agent class in the caller's
package can wire them without duplicating their logic.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.local_annotation_agent.backend import (
    LocalVLMConfig,
    get_or_start_service,
)
from app.annotation_providers.tool_surface import (
    AnnotationToolSurface,
    ToolAgentCapture,
    annotation_openai_tool_schemas,
    build_tool_agent_system_prompt,
    tool_agent_response,
    tool_agent_stage,
)
from app.shared.contracts import (
    AgentRequest,
    AgentResponse,
    StepEvent,
)
from app.local_annotation_agent.evaluators import (
    AttachmentPool,
    EvalPipelineResult,
    PipelineAttachment,
    _eval_llm_holder,
    render_inputs_for_prompt,
    run_evaluator_suite,
    set_active_attachments,
    set_active_stage,
    set_eval_llm,
    set_step_event_callback,
    set_vlm_chat_with_tools,
)
from app.local_annotation_agent.framework import (
    AgentState,
)

# Side-effect import: registers non-visual sub-agents with the framework.
import app.local_annotation_agent.sub_agents  # noqa: F401

LOGGER = logging.getLogger(__name__)

_LOCAL_NODE = "local_vlm_tool_agent"


class PlannerFormatError(RuntimeError):
    """Raised when the local planner returns no valid JSON step plan."""


# ---------------------------------------------------------------------------
# Plan parsing — caller's planner prompt drives the available agent menu
# ---------------------------------------------------------------------------


def _parse_planner_steps(plan_text: str) -> List[Dict[str, Any]]:
    """Parse planner VLM output into structured step dicts.

    Accepts ``{"steps": [{"step_id": int, "agent": str, "description": str,
    "requested_graphs": [...], "tools": [...]}, ...]}``. Unknown agent
    names are passed through — ``delegate_step`` will raise a clear error
    if the agent isn't registered. The framework no longer appends any
    step; the caller's planner prompt is the sole source of plan content.
    """
    steps_raw: Optional[list] = None
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", plan_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            brace_match = re.search(r"(\{.*\})", plan_text, re.DOTALL)
            parsed = json.loads(brace_match.group(1)) if brace_match else None
        if parsed and isinstance(parsed, dict) and "steps" in parsed:
            steps_raw = parsed["steps"]
    except (json.JSONDecodeError, ValueError):
        steps_raw = None

    if not steps_raw or not isinstance(steps_raw, list):
        raise PlannerFormatError(
            "Planner response did not contain a JSON object with a non-empty "
            "`steps` array."
        )

    structured: List[Dict[str, Any]] = []
    for i, raw_step in enumerate(steps_raw, start=1):
        if not isinstance(raw_step, dict):
            continue
        step_id = raw_step.get("step_id", i)
        agent_id = raw_step.get("agent") or raw_step.get("solver")
        if not agent_id:
            LOGGER.warning(
                "Step %s missing 'agent' field — skipping.", step_id,
            )
            continue
        desc = raw_step.get("description", f"Step {step_id}")
        req_graphs = raw_step.get("requested_graphs") or []
        tools = raw_step.get("tools") or []
        if not isinstance(req_graphs, list):
            req_graphs = []
        if not isinstance(tools, list):
            tools = []

        structured.append({
            "step_id": step_id,
            "agent": agent_id,
            "description": desc,
            "requested_graphs": req_graphs,
            "tools": tools,
        })

    if not structured:
        raise PlannerFormatError(
            "Planner response contained `steps`, but none had an `agent` field."
        )

    return structured


# ---------------------------------------------------------------------------
# Default node implementations — exported so a caller-defined root Agent
# can wire them. They read ``planner_prompt`` / ``synth_prompt`` from state
# (the runner seeds those from AgentRequest before invoking the graph).
# ---------------------------------------------------------------------------


def default_planner_node(state: AgentState) -> Dict[str, Any]:
    """Send the caller's planner_prompt to the VLM, parse the resulting plan."""
    planner_prompt = state.get("planner_prompt")
    if not planner_prompt:
        raise RuntimeError(
            "local runner: state['planner_prompt'] missing. The runner "
            "must seed it from AgentRequest before invoking the graph."
        )

    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    set_active_stage("planner", "main")
    set_active_attachments([])
    vlm_fn = _eval_llm_holder.get("vlm")
    if vlm_fn:
        raw_plan = vlm_fn(planner_prompt)
    else:
        raw_plan = "[VLM not available — using passthrough plan]"

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=planner_prompt,
        parent_output_text=raw_plan,
        parent_inputs=[],
        step_name="planner",
        parent_start=parent_start,
        parent_end=parent_end,
        evaluators=["format_evaluator"],  # plan is JSON; evidence check N/A
    )
    evaluated_plan = suite_result.final_result

    parsed_steps = _parse_planner_steps(evaluated_plan)

    # Seed the pool with the caller-provided initial attachments + the
    # planner's own plan text.
    initial_pool: Dict[str, PipelineAttachment] = {}
    for att in state.get("initial_attachments", []) or []:
        if isinstance(att, PipelineAttachment):
            initial_pool[att.name] = att
    plan_attachment = PipelineAttachment(
        name="planner.plan",
        kind="text",
        label="Planner Plan",
        content=evaluated_plan,
    )
    initial_pool[plan_attachment.name] = plan_attachment

    messages = list(state.get("messages", []))
    messages.append({"role": "planner", "content": evaluated_plan})

    return {
        "plan": evaluated_plan,
        "plan_steps": parsed_steps,
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "attachment_pool": initial_pool,
        "messages": messages,
    }


def _call_vlm(prompt: str, graph_image_bytes: List[bytes]) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)


def default_synth_node(state: AgentState) -> Dict[str, Any]:
    """Build the synth prompt, send it to the VLM, run the evaluator suite.

    Attachment picker is generic: render every ``init.*`` (caller-seeded
    inputs) and every ``step_solver.*`` (sub-agent outputs) attachment in
    name order. ``planner.*`` and ``partial.*`` are excluded — those are
    bookkeeping, not synth context.
    """
    synth_prompt_fn = state.get("synth_prompt")
    if not callable(synth_prompt_fn):
        raise RuntimeError(
            "local runner: state['synth_prompt'] must be a callable "
            "(state) -> (intro, outro). The runner seeds it from "
            "AgentRequest.synth_prompt before invoking the graph."
        )
    synth_intro, synth_outro = synth_prompt_fn(state)
    if not synth_intro or not synth_outro:
        raise RuntimeError(
            "local runner: synth_prompt callable returned empty intro/outro."
        )

    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    pool: AttachmentPool = state.get("attachment_pool", {})

    # Generic picker: caller inputs + sub-agent outputs, in stable order.
    parent_inputs: List[PipelineAttachment] = []
    for name in sorted(pool.keys()):
        if name.startswith("init."):
            parent_inputs.append(pool[name])
    for name in sorted(pool.keys()):
        if name.startswith("step_solver."):
            parent_inputs.append(pool[name])

    context_block = render_inputs_for_prompt(parent_inputs)

    vlm_prompt = "\n\n".join([synth_intro, context_block, synth_outro])
    eval_prompt = "\n\n".join([synth_intro, synth_outro])

    set_active_stage("synthesizer", "main")
    set_active_attachments(parent_inputs)
    raw_response = _call_vlm(vlm_prompt, [])
    if not raw_response:
        raise RuntimeError(
            f"local runner synthesizer: VLM returned empty response "
            f"(range=[{parent_start}, {parent_end}])"
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=eval_prompt,
        parent_output_text=raw_response,
        parent_inputs=parent_inputs,
        step_name="synthesizer",
        parent_start=parent_start,
        parent_end=parent_end,
    )
    evaluated_response = suite_result.final_result

    proposal_attachment = PipelineAttachment(
        name="synthesizer.response",
        kind="text",
        label="Synthesizer Output",
        content=evaluated_response,
    )

    messages.append({"role": "assistant", "content": evaluated_response})

    return {
        "evaluation": suite_result.final_verdict,
        "final_synth_response": evaluated_response,
        "attachment_pool": {proposal_attachment.name: proposal_attachment},
        "messages": messages,
    }


def default_eval_node(state: AgentState) -> Dict[str, Any]:
    return {"evaluation": state.get("evaluation", "pass")}


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------


def _local_vlm_config_from_request(request: AgentRequest) -> LocalVLMConfig:
    opts = request.config.provider_options
    return LocalVLMConfig(
        gguf_path=opts.get("gguf_path") or None,
        mmproj_path=opts.get("mmproj_path") or None,
        context_size=int(opts.get("context_size") or 32768),
        n_gpu_layers=int(opts.get("n_gpu_layers") if opts.get("n_gpu_layers") is not None else -1),
        hf_repo=request.config.model or str(opts.get("hf_repo") or "Qwen/Qwen2.5-VL-72B-Instruct"),
        quantization_type=str(opts.get("quantization_type") or "Q4_K_M"),
    )


def _wire_local_vlm(
    request: AgentRequest,
    step_events: List[StepEvent],
) -> None:
    """Bind the local VLM callables every sub-agent + evaluator reads."""
    vlm_service = get_or_start_service(_local_vlm_config_from_request(request))

    cb = request.callbacks

    def vlm_generate(prompt: str, images: Optional[List[bytes]] = None) -> str:
        from app.local_annotation_agent.evaluators import get_active_stage
        if cb.vlm_prompt:
            cb.vlm_prompt(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=images,
            max_tokens=request.config.max_new_tokens,
            temperature=request.config.temperature,
            stream_callback=cb.vlm_stream,
            reasoning_callback=cb.vlm_reasoning,
        )

    def llm_generate(prompt: str) -> str:
        from app.local_annotation_agent.evaluators import get_active_stage
        if cb.vlm_prompt:
            cb.vlm_prompt(prompt, get_active_stage())
        return vlm_service.generate(
            prompt,
            images=None,
            max_tokens=request.config.max_new_tokens,
            temperature=0.1,
            stream_callback=cb.vlm_stream,
            reasoning_callback=cb.vlm_reasoning,
        )

    def vlm_chat_with_tools(
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_handler: Callable[[str, Dict[str, Any]], str],
        images: Optional[List[bytes]] = None,
    ) -> str:
        from app.local_annotation_agent.evaluators import get_active_stage
        if cb.vlm_prompt:
            cb.vlm_prompt(prompt, get_active_stage())
        return vlm_service.chat_with_tools(
            prompt,
            tools=tools,
            tool_handler=tool_handler,
            images=images,
            max_tokens=request.config.max_new_tokens,
            temperature=request.config.temperature,
            stream_callback=cb.vlm_stream,
            reasoning_callback=cb.vlm_reasoning,
        )

    set_eval_llm(vlm_generate, llm_generate)
    set_vlm_chat_with_tools(vlm_chat_with_tools)

    # Bridge step_event callbacks both to the caller and into the
    # AgentResponse transcript.
    def step_event_bridge(summary: str, stage: Dict[str, Any]) -> None:
        step_events.append(StepEvent(
            stage=stage.get("node_name", "") if isinstance(stage, dict) else "",
            summary=summary,
            detail=dict(stage) if isinstance(stage, dict) else {},
        ))
        if cb.step_event:
            cb.step_event(summary, stage)

    set_step_event_callback(step_event_bridge)


def _extract_direct_json_payload(raw: str) -> Tuple[str, Dict[str, Any]] | None:
    """Return a JSON object emitted as plain assistant text, if present."""

    def _loads_object(text: str) -> Dict[str, Any] | None:
        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed.strip())
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    text = str(raw or "").strip()
    if not text:
        return None

    parsed = _loads_object(text)
    if parsed is not None:
        return json.dumps(parsed), parsed

    if "```json" in text:
        fenced = text.split("```json", 1)[1].split("```", 1)[0]
        parsed = _loads_object(fenced)
        if parsed is not None:
            return json.dumps(parsed), parsed
    elif "```" in text:
        fenced = text.split("```", 1)[1].split("```", 1)[0]
        parsed = _loads_object(fenced)
        if parsed is not None:
            return json.dumps(parsed), parsed

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        parsed = _loads_object(brace_match.group())
        if parsed is not None:
            return json.dumps(parsed), parsed

    return None


def _capture_direct_submission(capture: ToolAgentCapture) -> None:
    if capture.submitted:
        return
    extracted = _extract_direct_json_payload("".join(capture.text_chunks))
    if extracted is None:
        return
    payload_json, parsed = extracted
    if not any(key in parsed for key in ("label_ids", "proposals")):
        return
    capture.submit_payload = payload_json
    capture.submit_summary = str(
        parsed.get("reasoning") or parsed.get("summary") or ""
    )
    capture.submitted = True


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_local(request: AgentRequest) -> AgentResponse:
    """Execute one run on the local llama.cpp-backed tool-agent backend."""
    service = get_or_start_service(_local_vlm_config_from_request(request))
    capture = ToolAgentCapture(
        node_name=_LOCAL_NODE,
        cur_start=int(request.parent_start),
        cur_end=int(request.parent_end),
    )
    surface = AnnotationToolSurface(request, capture)
    cb = request.callbacks

    if cb.vlm_prompt:
        cb.vlm_prompt(request.planner_prompt, tool_agent_stage(_LOCAL_NODE, "main"))
    if cb.progress:
        cb.progress(_LOCAL_NODE, "session starting")

    def _call_tool(name: str, args: Dict[str, Any]) -> str:
        capture.tool_calls += 1
        if cb.progress:
            cb.progress(_LOCAL_NODE, f"tool {capture.tool_calls}: {name}")
        _result, text, images = surface.call_tool(name, args)
        for img_b64 in images:
            try:
                capture.rendered_images.append(base64.b64decode(img_b64))
            except (ValueError, TypeError):
                LOGGER.warning("local tool-agent: invalid image payload from %s", name)
        return text

    raw_response = service.chat_with_tools(
        request.planner_prompt,
        tools=annotation_openai_tool_schemas(request),
        tool_handler=_call_tool,
        max_tokens=request.config.max_new_tokens,
        temperature=request.config.temperature,
        max_rounds=int(request.config.provider_options.get("max_turns") or 30),
        stream_callback=cb.vlm_stream,
        reasoning_callback=cb.vlm_reasoning,
        system_prompt=build_tool_agent_system_prompt(request),
    )
    if raw_response:
        capture.text_chunks.append(raw_response)

    _capture_direct_submission(capture)

    if cb.progress:
        cb.progress(
            _LOCAL_NODE,
            f"done - {capture.tool_calls} tool call(s), submitted={capture.submitted}",
        )
    return tool_agent_response(capture, request)
