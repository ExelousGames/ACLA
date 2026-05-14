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

from app.services.llm.step_evaluator_agents import (
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

    Domain-specific fields (df_ref, parent_main_labels, ...) ride alongside
    framework fields (depth, call_stack, ...) — TypedDict total=False makes
    every key optional so each Agent uses what it needs.
    """
    # Goal + scope from parent
    goal: str
    parent_start: int
    parent_end: int

    # Pipeline scaffolding inherited from parent
    df_ref: Any
    parent_main_labels: list
    existing_children: list
    segment_data: dict
    available_labels: dict
    messages: Annotated[list, append_list]

    # Solver-specific inputs forwarded from the parent's plan step.
    # Must be declared here so LangGraph keeps them in the subgraph state —
    # undeclared keys are stripped on child.invoke().
    requested_graphs: list
    tools: list

    # Planner output
    plan: str
    plan_steps: list
    current_step_index: int
    step_results: Annotated[list, operator.add]
    all_graph_images: list
    all_graph_descriptions: list

    # Shared attachment pool
    attachment_pool: Annotated[Dict[str, PipelineAttachment], merge_pool]

    # Recursion bookkeeping.  call_stack is set once per invocation by
    # spawn() and never written during a subgraph's run, so it has no
    # reducer — overwrite semantics avoid accidental double-counting.
    # spawn_log uses append_list so each executor only emits its new
    # entries and the framework concatenates them into the state.
    depth: int
    call_stack: list
    spawn_log: Annotated[list, append_list]
    total_spawns: int

    # Evaluator output
    evaluation: str

    # Label-verifier / synthesizer state (carried for compatibility with
    # the outer pipeline; leaf agents leave these untouched)
    verified_labels: list
    verified_label_reasoning: dict
    final_labels: list
    final_label_annotations: list
    final_reasoning: str
    final_sub_start: int
    final_sub_end: int


# ---------------------------------------------------------------------------
# Spec + budget
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    """One agent: planner → step_solvers → synthesizer → evaluator.

    Every callable except executor_fn is optional. Missing planner means
    the framework emits a trivial single-step plan. Missing synthesizer
    means multi-step results are not merged (the evaluator sees the last
    solver's output). Missing evaluator means the subgraph terminates
    after the executor / synthesizer.

    Prefer subclassing :class:`Agent` to declare new agents — the base
    class enforces the four-phase contract at class-definition time and
    compiles down to an AgentSpec via :meth:`Agent.register`.
    """
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
        executor      — run ONE plan step; return a state delta. Almost
                        always either ``delegate_step(state, step, registry)``
                        for orchestrators or the terminal logic for leaves.
        synthesizer   — merge multi-step results into a single output, or
                        None to skip (evaluator sees the last solver's output).
        evaluator     — produce a final verdict / parse, or None to skip.

    Class-level metadata:
        name          — registry key. REQUIRED.
        consumes      — attachment names this agent reads from its sliced
                        pool. Supports fnmatch wildcards (e.g.
                        ``"step_solver.*.observations"``).
        produces      — attachment leaf names this agent commits to emit.
                        Used by the parent's renamespace step.
        delegates_to  — registered agent names this executor may spawn.

    Failing to override a phase method, OR forgetting to set ``name``,
    emits a warning at subclass creation time. Override a method and
    return ``None`` / ``{}`` to explicitly mark a phase as "skip"
    without triggering the warning.
    """

    name: str = ""
    consumes: List[str] = []
    produces: List[str] = []
    delegates_to: List[str] = []

    _PHASE_METHODS = ("planner", "executor", "synthesizer", "evaluator")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip abstract intermediate classes that explicitly opt out.
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

    # --- Phase methods (override these in subclasses) ----------------------

    def planner(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Decide the plan for this invocation.

        Return ``{"plan_steps": [...], "plan": "..."}`` — or any subset
        that updates state. Returning ``None`` (or an empty dict) tells
        the framework to fall back to a trivial single-step plan derived
        from ``state["goal"]``.
        """
        return None

    def executor(
        self,
        state: AgentState,
        step: Dict[str, Any],
        registry: "AgentRegistry",
    ) -> Optional[Dict[str, Any]]:
        """Run ONE plan step. Return a state delta.

        Orchestrator agents should ``return delegate_step(state, step,
        registry)`` to spawn the sub-agent named in ``step['agent']``.
        Leaf agents do the work inline and return their attachments.
        """
        return None

    def synthesizer(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Produce the agent's single canonical output from step results.

        Always runs when defined — even with a single plan step. Two jobs:

        1. **Merge** — when multiple steps ran, weave their per-step
           outputs into one cohesive result (e.g. multi-zoom prose →
           one paragraph per graph).
        2. **Re-frame to the agent's own scope** — step executors often
           run at a narrower scope than the agent itself (e.g. a zoom
           step's prompt has ``Index Range: [zoom_start, zoom_end]``,
           a sub-range of the parent). The synthesizer rewrites the
           output at the agent's full ``[parent_start, parent_end]`` so
           the next caller upstream sees a uniform, parent-scoped
           result regardless of how the work got split internally.

        Return ``None`` to skip (the evaluator will then see the last
        step's output verbatim, scope and all).
        """
        return None

    def evaluator(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Validate or finalize the output. Return ``None`` to skip."""
        return None

    # --- Registration ------------------------------------------------------

    @classmethod
    def register(cls) -> AgentSpec:
        """Compile this Agent class into an AgentSpec and register it.

        Phase methods that the subclass did not override are recorded as
        ``None`` in the spec, so the framework strips them out of the
        compiled subgraph entirely. Overridden methods that return None
        at runtime are tolerated by the framework (it falls back to a
        trivial plan for the planner; treats the others as no-ops).
        """
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
    max_total_spawns: int = 16
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


# Module-level default budget; pipeline runner may overwrite at startup.
DEFAULT_BUDGET = AgentBudget()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


AgentRegistry = Dict[str, Any]   # name -> compiled LangGraph
AGENT_REGISTRY: AgentRegistry = {}
AGENT_SPECS: Dict[str, AgentSpec] = {}


def register_agent(spec: AgentSpec) -> None:
    """Compile a spec and register it under its name."""
    AGENT_SPECS[spec.name] = spec
    AGENT_REGISTRY[spec.name] = build_agent_graph(spec)


def get_registered(name: str):
    """Lookup helper. Raises if the name is not registered."""
    if name not in AGENT_REGISTRY:
        raise RuntimeError(
            f"unknown agent '{name}'. registered: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[name]


# ---------------------------------------------------------------------------
# Pool slicing & renamespacing — the boundary check on every spawn
# ---------------------------------------------------------------------------


def slice_pool_for(
    pool: AttachmentPool,
    consumes: List[str],
) -> AttachmentPool:
    """Return only the attachments the child Agent is allowed to see.

    Each ``consumes`` entry is either an exact attachment name or an
    fnmatch-style pattern containing ``*`` (e.g. ``step_solver.*.observations``).
    Exact names always match; patterns match any pool key satisfying the glob.
    """
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
    """Rename a child's produced attachments under the parent's namespace.

    Keys are matched by their last segment (`.observations`, `.graph_images`,
    ...) against the child spec's `produces` list. Internal scratch
    (e.g. `planner.plan`) is dropped.
    """
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
    """Refuse a spawn whose (agent, start, end) is already on the call stack.

    Checks the *child's* triple against the parent's stack — catches the case
    where an ancestor is already running ``agent_name`` over the same range,
    e.g. a zoom of a zoom on the same indices.
    """
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
    """Invoke a sub-agent with depth / stack / budget / cycle bookkeeping.

    Returns the sub-agent's final state. Caller is responsible for
    renamespacing the produced attachments before merging into its own pool.
    """
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

    # Append a spawn-log row so the run is replayable / auditable.
    log_row = {
        "parent_step_id": parent_step_id,
        "child_agent": agent_name,
        "child_range": [sub_state.get("parent_start"), sub_state.get("parent_end")],
        "child_goal": sub_state.get("goal"),
        "child_verdict": result.get("evaluation"),
        "depth": sub_state["depth"],
    }
    # The log lives in the result so the caller can carry it forward via
    # the append_list reducer when merging back into the parent state.
    result_log = list(result.get("spawn_log", [])) + [log_row]
    result["spawn_log"] = result_log

    return result


# ---------------------------------------------------------------------------
# Generic delegate executor — used by orchestrator agents
# ---------------------------------------------------------------------------


# Step fields the framework handles itself — anything else in the step
# dict is forwarded into the sub-state for the child to read.
_RESERVED_STEP_FIELDS = {
    "step_id", "agent", "description", "goal",
    "parent_start", "parent_end",
}

# Child-state keys that delegate_step must NOT propagate back into the
# parent's state. These either belong to the child's own scope (goal,
# bounds, plan_steps) or to the calling agent's scaffolding (df_ref,
# parent_main_labels) that the child should not be able to overwrite.
# attachment_pool is handled separately (renamespaced).
_NON_PROPAGATED_CHILD_FIELDS = {
    "depth", "call_stack",
    "goal", "parent_start", "parent_end",
    "plan", "plan_steps", "current_step_index",
    "df_ref", "segment_data",
    "parent_main_labels", "existing_children", "available_labels",
    "attachment_pool",
    # Concatenation fields handled with explicit parent + child merge.
    "all_graph_images", "all_graph_descriptions",
}


def delegate_step(
    state: AgentState,
    step: Dict[str, Any],
    registry: AgentRegistry,
) -> Dict[str, Any]:
    """Spawn the sub-agent named in ``step['agent']`` and merge its outputs.

    The step dict must include ``agent`` and ``step_id``. Optional fields:
    ``parent_start``, ``parent_end``, ``goal`` / ``description``. Any other
    keys are forwarded verbatim into the child's state so solver-specific
    fields (e.g. ``requested_graphs``, ``tools``) reach the executor.

    Produced attachments are namespaced under ``step_solver.{step_id}.*``
    using the child spec's ``produces`` list.
    """
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
        "parent_main_labels": state.get("parent_main_labels", []),
        "existing_children": state.get("existing_children", []),
        "parent_start": step.get("parent_start", state.get("parent_start", 0)),
        "parent_end": step.get("parent_end", state.get("parent_end", 0)),
        "available_labels": state.get("available_labels", {}),
        "segment_data": state.get("segment_data", {}),
        "goal": step.get("goal") or step.get("description", "") or agent_name,
        "attachment_pool": slice_pool_for(parent_pool, spec.consumes),
        "messages": [],
        "all_graph_images": [],
        "all_graph_descriptions": [],
    }
    # Forward any solver-specific fields straight through.
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

    # Propagate the child's state fields except framework-internal ones
    # (see _NON_PROPAGATED_CHILD_FIELDS). Reducer-managed fields like
    # messages / step_results / spawn_log are forwarded as-is; the parent
    # state graph's reducers concatenate them with the parent's prior value.
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
    """Compile an AgentSpec into a uniform LangGraph subgraph.

    Topology: planner → executor (self-loops per step) → [synthesizer] →
    [evaluator] → END. Synthesizer runs whenever ``spec.synthesizer_fn``
    is set (even for single-step plans, so it can re-frame per-step
    output back to the agent's full scope); evaluator runs only when
    ``spec.evaluator_fn`` is set.
    """
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
        # An overriding planner that returns None / no plan_steps is
        # treated the same as "no planner needed" — fall back to a
        # trivial single-step plan so the executor still has something
        # to run.
        if not delta.get("plan_steps"):
            delta = {**_trivial_plan(), **delta}
        # Reset step counter at the start of every Agent invocation so a
        # re-used compiled graph doesn't inherit a stale index.
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
        # Only added to the graph when spec.synthesizer_fn is set.
        return spec.synthesizer_fn(state) or {}

    def evaluator_node(state: AgentState) -> Dict[str, Any]:
        # Only added to the graph when spec.evaluator_fn is set.
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
