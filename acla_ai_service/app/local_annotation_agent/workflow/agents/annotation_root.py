"""AnnotationRoot — the root Agent driving the local-backend annotation run.

The agent box ships generic planner / synth / eval node implementations
in ``agent/runners/local.py``. This module wires them into an Agent
subclass that declares which sub-agents the planner may delegate to.

Sub-agents in ``delegates_to``:
  * ``label_verifier`` — embedding-similarity filter (in the box).

Side-effect registration on import populates the box's AGENT_REGISTRY.
"""

from __future__ import annotations

from typing import Any, Dict

from app.local_annotation_agent.framework import Agent, AgentState, delegate_step
from app.local_annotation_agent.runner import (
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
    delegates_to = ["label_verifier"]

    def planner(self, state: AgentState) -> Dict[str, Any]:
        return default_planner_node(state)

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        if step.get("agent") in {"describe_graphs", "zoom"}:
            raise RuntimeError(
                "visual graph inspection agents are disabled for AI annotation"
            )
        return delegate_step(state, step, registry)

    def synthesizer(self, state: AgentState) -> Dict[str, Any]:
        return default_synth_node(state)

    def evaluator(self, state: AgentState) -> Dict[str, Any]:
        return default_eval_node(state)


ANNOTATION_ROOT_SPEC = AnnotationRoot.register()
