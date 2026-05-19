"""
Recursive Agent framework — uniform planner → step_solvers → synthesizer → evaluator.

Every Agent compiles to its own LangGraph subgraph following the SAME shape:

    planner (optional) ─► executor (loops per plan step) ─┬─► synthesizer ─┐
                                                          │   (if >1 step) │
                                                          │                ▼
                                                          └────────────► evaluator ─► END
                                                                          (optional)

- planner_fn=None  → framework substitutes a trivial single-step plan.
- synthesizer_fn=None → synthesizer bypassed; otherwise it always runs so
  it can normalize the final output (e.g. rewrite per-step ranges back to
  the parent range), even when there's only one plan step.
- evaluator_fn=None → straight to END.

A step solver is itself an Agent: ``executor_fn`` either runs the leaf
operation inline OR delegates to a registered sub-agent via the generic
``delegate_step`` helper. Recursion bottoms out at agents whose executor
does the terminal work directly.

Boundary contracts on each AgentSpec:
    consumes      attachment names the child needs from the parent's pool
    produces      attachment names the child commits to emit back
    delegates_to  agent names this executor is allowed to spawn

Recursion is bounded by AgentBudget and cycle detection on
(agent_name, parent_start, parent_end) triples carried in state['call_stack'].
"""

from __future__ import annotations

import fnmatch
import logging
import operator
from dataclasses import dataclass, field
from typing import Any, Annotated, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.agents.evaluators import (
    AttachmentPool,
    PipelineAttachment,
    merge_pool,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def append_list(left: Optional[list], right: Optional[list]) -> list:
    """List reducer — concat left + right, treating None as empty."""
    return list(left or []) + list(right or [])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """State flowing through any Agent subgraph.

    All fields are framework concerns: planner/executor/synthesizer
    plumbing, recursion bookkeeping, and the attachment pool every
    sub-agent reads and writes through. Domain inputs/outputs ride as
    attachments in ``attachment_pool``, not as named state fields.

    LangGraph filters keys not declared here out of the initial state and
    out of node deltas, so anything the runner seeds or any node returns
    MUST appear in this declaration.
    """
    goal: str
    parent_start: int
    parent_end: int

    df_ref: Any
    segment_data: dict
    messages: Annotated[list, append_list]

    # Caller-provided prompts + initial pool seeds — read by the planner /
    # synthesizer nodes that the runner wires.
    planner_prompt: str
    synth_prompt: Any                  # Callable[[state], tuple[str, str]]
    initial_attachments: list
    session_id: str

    requested_graphs: list
    tools: list
    question: str
    context: str
    questions: list
    graph_builds: dict

    plan: str
    plan_steps: list
    current_step_index: int
    step_results: Annotated[list, operator.add]
    all_graph_images: list
    all_graph_descriptions: list

    attachment_pool: Annotated[Dict[str, PipelineAttachment], merge_pool]

    depth: int
    call_stack: list
    spawn_log: Annotated[list, append_list]
    total_spawns: int

    evaluation: str
    final_synth_response: str


# ---------------------------------------------------------------------------
# Spec + budget
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    """One agent: planner → step_solvers → synthesizer → evaluator."""
    name: str
    executor_fn: Callable[[AgentState, Dict[str, Any], "AgentRegistry"], Dict[str, Any]]
    planner_fn: Optional[Callable[[AgentState], Dict[str, Any]]] = None
    synthesizer_fn: Optional[Callable[[AgentState], Dict[str, Any]]] = None
    evaluator_fn: Optional[Callable[[AgentState], Dict[str, Any]]] = None
    consumes: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    delegates_to: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent base class — enforces the four-phase contract
# ---------------------------------------------------------------------------


class Agent:
    """Base class for declaring an Agent. Subclasses MUST override the
    four phase methods.

    Required overrides (the contract):
        planner       — return a plan_steps delta, or None to use the
                        framework's trivial single-step plan.
        executor      — run ONE plan step; return a state delta.
        synthesizer   — merge multi-step results into a single output, or
                        None to skip.
        evaluator     — produce a final verdict / parse, or None to skip.

    Class-level metadata:
        name          — registry key. REQUIRED.
        consumes      — attachment names this agent reads from its sliced
                        pool. Supports fnmatch wildcards.
        produces      — attachment leaf names this agent commits to emit.
        delegates_to  — registered agent names this executor may spawn.
    """

    name: str = ""
    consumes: List[str] = []
    produces: List[str] = []
    delegates_to: List[str] = []

    _PHASE_METHODS = ("planner", "executor", "synthesizer", "evaluator")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "_agent_abstract", False):
            return

        missing: List[str] = []
        for method_name in cls._PHASE_METHODS:
            sub_method = getattr(cls, method_name, None)
            base_method = getattr(Agent, method_name)
            if sub_method is base_method:
                missing.append(method_name)
        if missing:
            LOGGER.warning(
                "Agent subclass '%s' did not override: %s. Each unoverridden "
                "phase will be skipped silently. Override the method and "
                "return None to mark a phase as 'skip' explicitly.",
                cls.__name__, missing,
            )
        if not cls.name:
            LOGGER.warning(
                "Agent subclass '%s' did not set the `name` class attribute "
                "— registration will fail.", cls.__name__,
            )

    def planner(self, state: AgentState) -> Optional[Dict[str, Any]]:
        return None

    def executor(
        self,
        state: AgentState,
        step: Dict[str, Any],
        registry: "AgentRegistry",
    ) -> Optional[Dict[str, Any]]:
        return None

    def synthesizer(self, state: AgentState) -> Optional[Dict[str, Any]]:
        return None

    def evaluator(self, state: AgentState) -> Optional[Dict[str, Any]]:
        return None

    @classmethod
    def register(cls) -> AgentSpec:
        """Compile this Agent class into an AgentSpec and register it."""
        if not cls.name:
            raise RuntimeError(
                f"Agent subclass '{cls.__name__}' has no `name` — "
                f"cannot register."
            )
        instance = cls()

        def _override_or_none(method_name: str) -> Optional[Callable]:
            sub_method = getattr(cls, method_name)
            base_method = getattr(Agent, method_name)
            if sub_method is base_method:
                return None
            return getattr(instance, method_name)

        spec = AgentSpec(
            name=cls.name,
            executor_fn=getattr(instance, "executor"),
            planner_fn=_override_or_none("planner"),
            synthesizer_fn=_override_or_none("synthesizer"),
            evaluator_fn=_override_or_none("evaluator"),
            consumes=list(cls.consumes),
            produces=list(cls.produces),
            delegates_to=list(cls.delegates_to),
        )
        register_agent(spec)
        return spec


@dataclass
class AgentBudget:
    """Recursion limits enforced at every spawn."""
    max_depth: int = 4
    max_total_spawns: int = 64
    max_react_rounds: int = 2

    def check(self, state: AgentState) -> None:
        d = state.get("depth", 0)
        if d > self.max_depth:
            raise RuntimeError(
                f"agent tree depth {d} exceeds budget {self.max_depth}"
            )
        t = state.get("total_spawns", 0)
        if t > self.max_total_spawns:
            raise RuntimeError(
                f"total spawn count {t} exceeds budget {self.max_total_spawns}"
            )


DEFAULT_BUDGET = AgentBudget()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


AgentRegistry = Dict[str, Any]
AGENT_REGISTRY: AgentRegistry = {}
AGENT_SPECS: Dict[str, AgentSpec] = {}


def register_agent(spec: AgentSpec) -> None:
    """Compile a spec and register it under its name."""
    AGENT_SPECS[spec.name] = spec
    AGENT_REGISTRY[spec.name] = build_agent_graph(spec)


def get_registered(name: str):
    if name not in AGENT_REGISTRY:
        raise RuntimeError(
            f"unknown agent '{name}'. registered: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[name]


# ---------------------------------------------------------------------------
# Pool slicing & renamespacing — boundary check on every spawn
# ---------------------------------------------------------------------------


def slice_pool_for(
    pool: AttachmentPool,
    consumes: List[str],
) -> AttachmentPool:
    """Return only attachments the child Agent is allowed to see."""
    out: AttachmentPool = {}
    for pattern in consumes:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            for name, att in pool.items():
                if fnmatch.fnmatchcase(name, pattern):
                    out[name] = att
        elif pattern in pool:
            out[pattern] = pool[pattern]
    return out


def renamespace(
    pool: AttachmentPool,
    prefix: str,
    produces: List[str],
) -> AttachmentPool:
    """Rename child's produced attachments under the parent's namespace."""
    out: AttachmentPool = {}
    produces_set = set(produces)
    for name, att in pool.items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in produces_set and name not in produces_set:
            continue
        new_name = f"{prefix}.{leaf}"
        out[new_name] = att.copy(update={"name": new_name})
    return out


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def check_cycle(
    parent_state: AgentState,
    agent_name: str,
    child_start: Any,
    child_end: Any,
) -> None:
    """Refuse a spawn whose (agent, start, end) is already on the call stack."""
    triple = (agent_name, child_start, child_end)
    stack = parent_state.get("call_stack", [])
    if triple in stack:
        raise RuntimeError(
            f"cycle detected: {agent_name} already on call stack for "
            f"range [{child_start}, {child_end}]"
        )


# ---------------------------------------------------------------------------
# Spawn — the single recursion primitive
# ---------------------------------------------------------------------------


def spawn(
    agent_name: str,
    sub_state: AgentState,
    parent_state: AgentState,
    budget: AgentBudget = DEFAULT_BUDGET,
    parent_step_id: Any = None,
) -> AgentState:
    """Invoke a sub-agent with depth / stack / budget / cycle bookkeeping."""
    child = get_registered(agent_name)

    sub_state = dict(sub_state)
    sub_state["depth"] = parent_state.get("depth", 0) + 1
    sub_state["call_stack"] = list(parent_state.get("call_stack", [])) + [
        (agent_name, sub_state.get("parent_start"), sub_state.get("parent_end"))
    ]
    sub_state["total_spawns"] = parent_state.get("total_spawns", 0) + 1

    budget.check(sub_state)
    check_cycle(
        parent_state,
        agent_name,
        sub_state.get("parent_start"),
        sub_state.get("parent_end"),
    )

    LOGGER.info(
        "spawn agent=%s depth=%d range=[%s,%s] total=%d",
        agent_name, sub_state["depth"],
        sub_state.get("parent_start"), sub_state.get("parent_end"),
        sub_state["total_spawns"],
    )

    result = child.invoke(sub_state, config={"recursion_limit": 100})

    log_row = {
        "parent_step_id": parent_step_id,
        "child_agent": agent_name,
        "child_range": [sub_state.get("parent_start"), sub_state.get("parent_end")],
        "child_goal": sub_state.get("goal"),
        "child_verdict": result.get("evaluation"),
        "depth": sub_state["depth"],
    }
    result_log = list(result.get("spawn_log", [])) + [log_row]
    result["spawn_log"] = result_log

    return result


# ---------------------------------------------------------------------------
# Generic delegate executor — used by orchestrator agents
# ---------------------------------------------------------------------------


_RESERVED_STEP_FIELDS = {
    "step_id", "agent", "description", "goal",
    "parent_start", "parent_end",
}

_NON_PROPAGATED_CHILD_FIELDS = {
    "depth", "call_stack",
    "goal", "parent_start", "parent_end",
    "plan", "plan_steps", "current_step_index",
    "df_ref", "segment_data",
    "attachment_pool",
    "all_graph_images", "all_graph_descriptions",
}


def delegate_step(
    state: AgentState,
    step: Dict[str, Any],
    registry: AgentRegistry,
) -> Dict[str, Any]:
    """Spawn the sub-agent named in ``step['agent']`` and merge its outputs."""
    agent_name = step.get("agent")
    if not agent_name:
        raise RuntimeError(
            f"delegate_step: step is missing 'agent' field: {step!r}"
        )
    step_id = step.get("step_id")
    if step_id is None:
        raise RuntimeError(
            f"delegate_step: step is missing 'step_id': {step!r}"
        )

    spec = AGENT_SPECS.get(agent_name)
    if spec is None:
        raise RuntimeError(
            f"delegate_step: unknown agent '{agent_name}'. "
            f"registered: {sorted(AGENT_SPECS)}"
        )

    parent_pool: AttachmentPool = state.get("attachment_pool", {})

    sub_state: Dict[str, Any] = {
        "df_ref": state.get("df_ref"),
        "parent_start": step.get("parent_start", state.get("parent_start", 0)),
        "parent_end": step.get("parent_end", state.get("parent_end", 0)),
        "segment_data": state.get("segment_data", {}),
        "goal": step.get("goal") or step.get("description", "") or agent_name,
        "attachment_pool": slice_pool_for(parent_pool, spec.consumes),
        "messages": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
    }
    for k, v in step.items():
        if k not in _RESERVED_STEP_FIELDS:
            sub_state[k] = v

    child_result = spawn(
        agent_name,
        sub_state,
        state,
        parent_step_id=step_id,
    )

    ns = f"step_solver.{step_id}"
    renamed = renamespace(
        child_result.get("attachment_pool", {}),
        ns,
        spec.produces,
    )

    delta: Dict[str, Any] = {
        k: v for k, v in child_result.items()
        if k not in _NON_PROPAGATED_CHILD_FIELDS
    }
    delta["attachment_pool"] = renamed
    delta["all_graph_images"] = (
        state.get("all_graph_images", [])
        + child_result.get("all_graph_images", [])
    )
    delta["all_graph_descriptions"] = (
        state.get("all_graph_descriptions", [])
        + child_result.get("all_graph_descriptions", [])
    )
    return delta


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent_graph(spec: AgentSpec):
    """Compile an AgentSpec into a uniform LangGraph subgraph."""
    graph = StateGraph(AgentState)

    def _trivial_plan() -> Dict[str, Any]:
        return {
            "plan": "(trivial single-step plan)",
            "plan_steps": [{
                "step_id": 1,
                "description": "(default step)",
            }],
            "current_step_index": 0,
        }

    def planner_node(state: AgentState) -> Dict[str, Any]:
        if spec.planner_fn is None:
            return _trivial_plan()
        delta = spec.planner_fn(state) or {}
        if not delta.get("plan_steps"):
            delta = {**_trivial_plan(), **delta}
        delta.setdefault("current_step_index", 0)
        return delta

    def executor_node(state: AgentState) -> Dict[str, Any]:
        plan_steps = state.get("plan_steps", [])
        idx = state.get("current_step_index", 0)
        if idx >= len(plan_steps):
            return {}
        step = plan_steps[idx]
        delta = spec.executor_fn(state, step, AGENT_REGISTRY) or {}
        delta["current_step_index"] = idx + 1
        return delta

    def synthesizer_node(state: AgentState) -> Dict[str, Any]:
        return spec.synthesizer_fn(state) or {}

    def evaluator_node(state: AgentState) -> Dict[str, Any]:
        return spec.evaluator_fn(state) or {}

    def after_executor_router(state: AgentState) -> str:
        plan_steps = state.get("plan_steps", [])
        idx = state.get("current_step_index", 0)
        if idx < len(plan_steps):
            return "executor"
        if spec.synthesizer_fn is not None:
            return "synthesizer"
        if spec.evaluator_fn is not None:
            return "evaluator"
        return "end"

    def after_synth_router(state: AgentState) -> str:
        if spec.evaluator_fn is not None:
            return "evaluator"
        return "end"

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    if spec.synthesizer_fn is not None:
        graph.add_node("synthesizer", synthesizer_node)
    if spec.evaluator_fn is not None:
        graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")

    executor_routes: Dict[str, Any] = {"executor": "executor", "end": END}
    if spec.synthesizer_fn is not None:
        executor_routes["synthesizer"] = "synthesizer"
    if spec.evaluator_fn is not None:
        executor_routes["evaluator"] = "evaluator"
    graph.add_conditional_edges(
        "executor", after_executor_router, executor_routes,
    )

    if spec.synthesizer_fn is not None:
        synth_routes: Dict[str, Any] = {"end": END}
        if spec.evaluator_fn is not None:
            synth_routes["evaluator"] = "evaluator"
        graph.add_conditional_edges(
            "synthesizer", after_synth_router, synth_routes,
        )

    if spec.evaluator_fn is not None:
        graph.add_edge("evaluator", END)

    return graph.compile()
