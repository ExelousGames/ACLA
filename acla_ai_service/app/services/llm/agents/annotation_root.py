"""
annotation_root Agent — the ONE annotation pipeline as a uniform Agent.

Topology (inherited from the framework):

    planner ──► step_solvers (loop) ──► synthesizer ──► evaluator ──► END
                  ▲                                       (no-op:
                  │                                        synthesizer
                  │                                        ran the eval
              describe_graphs                              suite inline)
              describe_graphs
              ...
              label_verifier        ◄── always appended as the last step

This is the only annotation agent. Callers parameterise it via state:

    planner_prompt       full text the planner sends to the VLM
    synth_prompt_intro   text the synthesizer prepends before the rendered
                         context block (skills + task framing)
    synth_prompt_outro   text the synthesizer appends after the rendered
                         context block (output-schema + hard rules)
    initial_attachments  list[PipelineAttachment] seeded into the pool before
                         the planner runs (e.g. ``init.parent_segment``)

    df_ref, parent_start, parent_end, parent_main_labels, ...   pre-existing
                         framework state fields. ``parent_main_labels`` drives
                         the label_verifier's candidate pool.

Output (in state):

    final_synth_response  raw evaluator-finalised VLM response from the
                          synthesizer. Pipeline callers parse this for their
                          flow-specific shape (sub-segment proposals vs.
                          lap-section revised range, etc.).

The agent itself has no knowledge of any specific output schema or flow.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.services.llm.agent_framework import (
    Agent,
    AgentState,
    delegate_step,
)
from app.services.llm.step_evaluator_agents import (
    AttachmentPool,
    EvalPipelineResult,
    PipelineAttachment,
    _eval_llm_holder,
    render_inputs_for_prompt,
    run_evaluator_suite,
    set_active_attachments,
    set_active_stage,
)

LOGGER = logging.getLogger(__name__)

ANNOTATION_ROOT_AGENT_NAME = "annotation_root"
DEFAULT_STEP_AGENT = "describe_graphs"


# ---------------------------------------------------------------------------
# Plan parsing helpers
# ---------------------------------------------------------------------------


def _parse_planner_steps(plan_text: str) -> List[Dict[str, Any]]:
    """Parse planner VLM output into structured step dicts.

    Each returned step dict has keys:
        step_id, agent, description, requested_graphs, tools
    """
    from app.services.llm.annotation_agent_tools import AGENT_GRAPH_DEFINITIONS

    all_graph_ids = [g["id"] for g in AGENT_GRAPH_DEFINITIONS]

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
        LOGGER.warning("Could not parse planner steps; using fallback.")
        return [{
            "step_id": 1,
            "agent": DEFAULT_STEP_AGENT,
            "description": "Analyse all telemetry graphs and propose the most fitting labels.",
            "requested_graphs": list(all_graph_ids),
            "tools": [],
        }]

    structured: List[Dict[str, Any]] = []
    for i, raw_step in enumerate(steps_raw, start=1):
        step_id = raw_step.get("step_id", i)
        desc = raw_step.get("description", f"Step {step_id}")
        # Accept legacy "solver" key from older planners.
        agent_id = raw_step.get("agent") or raw_step.get("solver") or DEFAULT_STEP_AGENT
        if agent_id != "describe_graphs":
            LOGGER.warning(
                "Step %s requested agent '%s'; only 'describe_graphs' is a "
                "valid planner-chosen agent. Falling back to describe_graphs.",
                step_id, agent_id,
            )
            agent_id = DEFAULT_STEP_AGENT

        req_graphs = raw_step.get("requested_graphs", [])
        if not req_graphs:
            desc_lower = desc.lower()
            req_graphs = [g for g in all_graph_ids
                          if g in desc_lower or g.replace("_", " ") in desc_lower]
        req_graphs = [g for g in req_graphs if g in all_graph_ids]

        tools = raw_step.get("tools", [])
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
# Planner — calls VLM with caller-provided prompt, appends label_verifier step
# ---------------------------------------------------------------------------


def _planner(state: AgentState) -> Dict[str, Any]:
    """Run the planner phase.

    Reads ``state['planner_prompt']`` and sends it to the VLM. The caller
    is responsible for including everything the VLM needs to make a plan
    (task framing, skill blocks, eligible graphs, output-format directive).

    Seeds the attachment pool with ``state['initial_attachments']`` so
    downstream step solvers can consume them (e.g. ``init.parent_segment``).
    """
    planner_prompt = state.get("planner_prompt")
    if not planner_prompt:
        raise RuntimeError(
            "annotation_root._planner: state['planner_prompt'] is required. "
            "The caller (pipeline wrapper) must build the full planner prompt."
        )

    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    set_active_stage("planner", "main")
    set_active_attachments([])
    vlm_fn = _eval_llm_holder.get("vlm")
    if vlm_fn:
        raw_plan = vlm_fn(planner_prompt)
    else:
        raw_plan = (
            "[VLM not available — using passthrough plan] "
            "Examine all telemetry features and propose the most fitting labels."
        )

    suite_result: EvalPipelineResult = run_evaluator_suite(
        parent_prompt=planner_prompt,
        parent_output_text=raw_plan,
        parent_inputs=[],
        step_name="planner",
        parent_start=parent_start,
        parent_end=parent_end,
    )
    evaluated_plan = suite_result.final_result

    parsed_steps = _parse_planner_steps(evaluated_plan)

    # Append label_verifier as the trailing step solver. The framework
    # always runs it after every describe_graphs step so the synthesizer
    # gets an embedding-filtered candidate shortlist.
    next_step_id = max((s.get("step_id", 0) for s in parsed_steps), default=0) + 1
    parsed_steps.append({
        "step_id": next_step_id,
        "agent": "label_verifier",
        "description": "Filter candidate labels by embedding similarity to step observations.",
    })

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


# ---------------------------------------------------------------------------
# Synthesizer — renders the pool + caller intros around it, calls VLM
# ---------------------------------------------------------------------------


def _call_vlm(prompt: str, graph_image_bytes: List[bytes]) -> str:
    vlm_fn = _eval_llm_holder.get("vlm")
    if not vlm_fn:
        return ""
    if graph_image_bytes:
        return vlm_fn(prompt, graph_image_bytes)
    return vlm_fn(prompt)


def _synthesizer(state: AgentState) -> Dict[str, Any]:
    """Run the synthesizer phase.

    Composes the final synthesizer prompt as::

        {synth_prompt_intro}

        {rendered context — init.parent_segment + verified_labels + observations}

        {synth_prompt_outro}

    Sends it to the VLM, runs the evaluator suite, and stashes the
    evaluator-finalised raw response under ``final_synth_response`` for
    the pipeline caller to parse.
    """
    synth_intro = state.get("synth_prompt_intro")
    synth_outro = state.get("synth_prompt_outro")
    # Allow callables so the caller can defer prompt construction until
    # synth time (e.g. to read the freshly emitted verified_labels from
    # the post-label_verifier state).
    if callable(synth_intro):
        synth_intro = synth_intro(state)
    if callable(synth_outro):
        synth_outro = synth_outro(state)
    if not synth_intro or not synth_outro:
        raise RuntimeError(
            "annotation_root._synthesizer: state['synth_prompt_intro'] and "
            "['synth_prompt_outro'] are required. The caller (pipeline "
            "wrapper) must build them."
        )

    messages = list(state.get("messages", []))
    parent_start = state.get("parent_start", 0)
    parent_end = state.get("parent_end", 0)

    pool: AttachmentPool = state.get("attachment_pool", {})

    # Collect context inputs in stable order: parent_segment → verified
    # labels → observations.
    parent_inputs: List[PipelineAttachment] = []
    if "init.parent_segment" in pool:
        parent_inputs.append(pool["init.parent_segment"])
    for name in sorted(pool.keys()):
        if name.endswith(".verified_labels"):
            parent_inputs.append(pool[name])
    for name in sorted(pool.keys()):
        if name.endswith(".observations"):
            parent_inputs.append(pool[name])

    context_block = render_inputs_for_prompt(parent_inputs)

    vlm_prompt = "\n\n".join([synth_intro, context_block, synth_outro])
    # The evaluator suite needs to see the prompt without the inlined
    # context (its evidence-evaluator inspects ``parent_inputs`` directly).
    eval_prompt = "\n\n".join([synth_intro, synth_outro])

    set_active_stage("synthesizer", "main")
    set_active_attachments(parent_inputs)
    raw_response = _call_vlm(vlm_prompt, [])
    if not raw_response:
        raise RuntimeError(
            f"annotation_root synthesizer: VLM returned empty response "
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


# ---------------------------------------------------------------------------
# Evaluator — no-op pass-through (synthesizer ran the suite inline)
# ---------------------------------------------------------------------------


def _evaluator(state: AgentState) -> Dict[str, Any]:
    return {"evaluation": state.get("evaluation", "pass")}


class AnnotationRoot(Agent):
    """The only annotation agent. Flow-agnostic.

    Contract: caller seeds ``planner_prompt``, ``synth_prompt_intro``,
    ``synth_prompt_outro``, ``initial_attachments``, plus the standard
    framework fields (``df_ref``, ``parent_start``, ``parent_end``,
    ``parent_main_labels``, ``existing_children``, ``available_labels``).
    The agent runs planner → describe_graphs+label_verifier → synthesizer
    and emits ``final_synth_response`` as a raw string for the caller to
    parse however its use case demands.
    """

    name = ANNOTATION_ROOT_AGENT_NAME
    consumes: list = []       # root: starts from raw state inputs
    produces = ["response"]
    delegates_to = ["describe_graphs", "label_verifier"]

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return _planner(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return delegate_step(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return _synthesizer(state)

    def evaluator(self, state: AgentState) -> Dict[str, Any]:
        return _evaluator(state)


ANNOTATION_ROOT_SPEC = AnnotationRoot.register()
