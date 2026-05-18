"""AnnotationRoot — the root Agent driving the local-backend annotation run.

The agent box ships generic planner / synth / eval node implementations
in ``agent/runners/local.py``. This module wires them into an Agent
subclass that declares which sub-agents the planner may delegate to.

Sub-agents in ``delegates_to``:
  * ``describe_graphs`` — generic telemetry-graph describer (in the box).
  * ``zoom`` — generic VLM-driven zoom worker (in the box).
  * ``label_verifier`` — embedding-similarity filter (in the box, peer
    of describe_graphs and zoom). Dispatched from the planner JSON plan.

Side-effect registration on import populates the box's AGENT_REGISTRY.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.llm.agent.framework import Agent, AgentState, delegate_step
from app.services.llm.agent.runners.local import (
    default_planner_node,
    default_synth_node,
    default_eval_node,
)

ANNOTATION_ROOT_AGENT_NAME = "annotation_root"


class AnnotationRoot(Agent):
    """Root Agent driving the local LangGraph execution for annotation."""

    name = ANNOTATION_ROOT_AGENT_NAME
    consumes: list = []
    produces = ["response"]
    delegates_to = ["describe_graphs", "label_verifier", "zoom"]

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return default_planner_node(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return delegate_step(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return default_synth_node(state)

    def evaluator(self, state: AgentState) -> Dict[str, Any]:
        return default_eval_node(state)


ANNOTATION_ROOT_SPEC = AnnotationRoot.register()
