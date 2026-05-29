"""
Local runner — drives a LangGraph subgraph backed by the local VLM service.

Topology (built by the framework from the root Agent the caller registers):

    planner ──► executor (loops per plan step) ──► synthesizer ──► evaluator ──► END

The runner:
  1. Wires the VLM/LLM callables into the shared eval-LLM holder so every
     sub-agent and evaluator picks them up.
  2. Seeds the initial graph state from the AgentRequest (df_ref, range,
     attachments, planner_prompt, synth_prompt, extra_state).
  3. Streams the compiled graph, capturing node events into a transcript.
  4. Returns an AgentResponse — the synthesiser's raw text plus every
     attachment/graph-image/message the run produced.

This runner contains NO domain logic. The planner sends ``planner_prompt``
verbatim to the VLM; what comes back is parsed as a JSON plan of
``{step_id, agent, description, requested_graphs, tools}`` steps and
dispatched to registered sub-agents. The caller chooses the root Agent
to invoke via ``AgentRequest.extra_state["root_agent"]``.

The phase helpers ``default_planner_node`` / ``default_synth_node`` /
``default_eval_node`` are exported so a root Agent class in the caller's
package can wire them without duplicating their logic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from app.local_annotation_agent.backend import (
    LocalVLMConfig,
    get_or_start_service,
)
from app.shared.contracts import (
    AgentRequest,
    AgentResponse,
    Attachment,
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
    AGENT_REGISTRY,
    AgentState,
)

# Side-effect import: registers describe_graphs and zoom with the framework.
import app.local_annotation_agent.sub_agents  # noqa: F401

LOGGER = logging.getLogger(__name__)


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
        LOGGER.warning(
            "Could not parse planner steps; falling back to a trivial "
            "single describe_graphs step over all available graphs.",
        )
        from app.shared.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS
        return [{
            "step_id": 1,
            "agent": "describe_graphs",
            "description": "Analyse all available graphs.",
            "requested_graphs": [g["id"] for g in AGENT_GRAPH_DEFINITIONS],
            "tools": [],
        }]

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


def _wire_local_vlm(
    request: AgentRequest,
    step_events: List[StepEvent],
) -> None:
    """Bind the local VLM callables every sub-agent + evaluator reads."""
    cfg = LocalVLMConfig(
        gguf_path=request.config.gguf_path,
        mmproj_path=request.config.mmproj_path,
        context_size=request.config.context_size,
        n_gpu_layers=request.config.n_gpu_layers,
        hf_repo=request.config.hf_repo,
        quantization_type=request.config.quantization_type,
    )
    vlm_service = get_or_start_service(cfg)

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


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_local(request: AgentRequest) -> AgentResponse:
    """Execute one run on the local LangGraph backend.

    The caller specifies which registered Agent to invoke as the root via
    ``request.extra_state["root_agent"]``. The runner has no built-in root
    name — that's the application's choice.
    """
    root_agent = str(request.extra_state.get("root_agent") or "").strip()
    if not root_agent:
        raise RuntimeError(
            "local runner: request.extra_state['root_agent'] must name a "
            "registered Agent. The agent box is domain-free; the caller "
            "chooses the root."
        )
    if root_agent not in AGENT_REGISTRY:
        raise RuntimeError(
            f"local runner: root_agent '{root_agent}' is not registered. "
            f"Registered agents: {sorted(AGENT_REGISTRY)}"
        )

    step_events: List[StepEvent] = []
    _wire_local_vlm(request, step_events)

    initial_state: Dict[str, Any] = {
        "df_ref": request.df_ref,
        "parent_start": int(request.parent_start),
        "parent_end": int(request.parent_end),
        "segment_data": {
            "session_id": request.session_id,
            "start_index": int(request.parent_start),
            "end_index": int(request.parent_end),
        },
        "plan": "",
        "plan_steps": [],
        "current_step_index": 0,
        "step_results": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
        "attachment_pool": {},
        "evaluation": "",
        "final_synth_response": "",
        "messages": [],
        "depth": 0,
        "call_stack": [],
        "spawn_log": [],
        "total_spawns": 0,
        # Caller-provided prompts + initial pool seeds.
        "planner_prompt": request.planner_prompt,
        "synth_prompt": request.synth_prompt,
        "initial_attachments": list(request.initial_attachments),
    }
    # Bag of arbitrary fields the caller passes through. Only keys that
    # are declared in the registered Agent's state schema actually survive
    # LangGraph's filtering; the rest are silently dropped.
    for k, v in request.extra_state.items():
        if k == "root_agent":
            continue
        if k not in initial_state:
            initial_state[k] = v

    cb = request.callbacks

    graph = AGENT_REGISTRY[root_agent]
    final_state: Dict[str, Any] = dict(initial_state)
    for event in graph.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue
            final_state.update(node_output)
            if cb.progress:
                cb.progress(node_name, _progress_detail(node_name, final_state))

    set_eval_llm(None, None)
    set_vlm_chat_with_tools(None)

    # Build the AgentResponse. Repack the framework's PipelineAttachments
    # into the contract's Attachment shape so the caller never sees the
    # internal model.
    attachments_out: Dict[str, Attachment] = {}
    pool: Dict[str, PipelineAttachment] = final_state.get("attachment_pool", {})
    for name, pa in pool.items():
        attachments_out[name] = Attachment(
            name=pa.name,
            kind=_translate_attachment_kind(pa.kind),
            label=pa.label,
            content=pa.content,
            content_schema=pa.content_schema or None,
        )

    return AgentResponse(
        raw_response=final_state.get("final_synth_response", ""),
        verdict=final_state.get("evaluation", ""),
        attachments=attachments_out,
        step_events=step_events,
        graph_images=list(final_state.get("all_graph_images", [])),
        plan_steps=list(final_state.get("plan_steps", [])),
        messages=list(final_state.get("messages", [])),
    )


def _translate_attachment_kind(kind: str) -> str:
    # PipelineAttachment uses ``image_set``; contract Attachment uses
    # ``image``. Map either way to the contract vocabulary.
    if kind == "image_set":
        return "image"
    return kind  # text, structured already match


def _progress_detail(node_name: str, state: Dict[str, Any]) -> str:
    """Human-readable progress line per node, surfaced via callbacks."""
    if node_name == "planner":
        return f"Planned {len(state.get('plan_steps', []))} step(s)"
    if node_name == "executor":
        idx = state.get("current_step_index", 0)
        total = len(state.get("plan_steps", []))
        steps = state.get("plan_steps", [])
        just_done = idx - 1
        if 0 <= just_done < len(steps):
            step = steps[just_done]
            return f"Ran step {idx}/{total} via agent '{step.get('agent', '?')}'"
        return f"Ran step {idx}/{total}"
    if node_name == "synthesizer":
        return "Synthesizer emitted final response"
    if node_name == "evaluator":
        return f"Verdict: {state.get('evaluation', '?')}"
    return ""
