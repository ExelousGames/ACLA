"""Unit tests for the agent framework's pure helpers.

Covers slice_pool_for (including fnmatch patterns), renamespace,
check_cycle, AgentBudget, and the zoom-decision parser in
describe_graphs. Does NOT exercise the LangGraph subgraph or the VLM —
those require the live service.
"""

from __future__ import annotations

import pytest

from app.agents.framework import (
    AGENT_REGISTRY,
    AGENT_SPECS,
    Agent,
    AgentBudget,
    check_cycle,
    renamespace,
    slice_pool_for,
)
from app.agents.sub_agents.describe_graphs import (
    MIN_ZOOM_SPAN,
    _parse_zoom_decision,
)
from app.agents.evaluators import PipelineAttachment


def _att(name: str, content: str = "x") -> PipelineAttachment:
    return PipelineAttachment(name=name, kind="text", label=name, content=content)


# ---------------------------------------------------------------------------
# slice_pool_for
# ---------------------------------------------------------------------------


def test_slice_pool_for_keeps_only_named() -> None:
    pool = {
        "init.parent_segment": _att("init.parent_segment"),
        "planner.plan": _att("planner.plan"),
        "step_solver.1.observations": _att("step_solver.1.observations"),
    }
    sliced = slice_pool_for(pool, ["init.parent_segment"])
    assert set(sliced.keys()) == {"init.parent_segment"}


def test_slice_pool_for_skips_missing() -> None:
    pool = {"init.parent_segment": _att("init.parent_segment")}
    sliced = slice_pool_for(pool, ["init.parent_segment", "not_there"])
    assert set(sliced.keys()) == {"init.parent_segment"}


def test_slice_pool_for_supports_wildcards() -> None:
    pool = {
        "init.parent_segment": _att("init.parent_segment"),
        "step_solver.1.observations": _att("step_solver.1.observations"),
        "step_solver.2.observations": _att("step_solver.2.observations"),
        "step_solver.2.graph_images": _att("step_solver.2.graph_images"),
        "planner.plan": _att("planner.plan"),
    }
    sliced = slice_pool_for(
        pool, ["init.parent_segment", "step_solver.*.observations"],
    )
    assert set(sliced.keys()) == {
        "init.parent_segment",
        "step_solver.1.observations",
        "step_solver.2.observations",
    }


def test_slice_pool_for_wildcard_matches_nothing_yields_empty() -> None:
    pool = {"planner.plan": _att("planner.plan")}
    sliced = slice_pool_for(pool, ["step_solver.*.observations"])
    assert sliced == {}


# ---------------------------------------------------------------------------
# renamespace
# ---------------------------------------------------------------------------


def test_renamespace_filters_by_produces() -> None:
    pool = {
        "step_solver.1.observations": _att("step_solver.1.observations"),
        "step_solver.1.graph_images": _att("step_solver.1.graph_images"),
        "planner.plan": _att("planner.plan"),   # not in produces — drop
    }
    out = renamespace(pool, "step_solver.3.zoom.1.2",
                      ["observations", "graph_images"])
    assert set(out.keys()) == {
        "step_solver.3.zoom.1.2.observations",
        "step_solver.3.zoom.1.2.graph_images",
    }


def test_renamespace_updates_inner_name_field() -> None:
    pool = {"step_solver.1.observations": _att("step_solver.1.observations")}
    out = renamespace(pool, "ns.prefix", ["observations"])
    att = out["ns.prefix.observations"]
    assert att.name == "ns.prefix.observations"


# ---------------------------------------------------------------------------
# check_cycle
# ---------------------------------------------------------------------------


def test_check_cycle_raises_on_duplicate_triple() -> None:
    parent_state = {
        "call_stack": [("describe_graphs", 100, 200)],
    }
    with pytest.raises(RuntimeError, match="cycle detected"):
        check_cycle(parent_state, "describe_graphs", 100, 200)


def test_check_cycle_allows_different_range() -> None:
    parent_state = {
        "call_stack": [("describe_graphs", 100, 200)],
    }
    check_cycle(parent_state, "describe_graphs", 100, 150)
    check_cycle(parent_state, "describe_graphs", 300, 400)


def test_check_cycle_allows_different_agent() -> None:
    parent_state = {
        "call_stack": [("describe_graphs", 100, 200)],
    }
    check_cycle(parent_state, "tabular_summary", 100, 200)


# ---------------------------------------------------------------------------
# AgentBudget
# ---------------------------------------------------------------------------


def test_budget_depth_limit() -> None:
    b = AgentBudget(max_depth=3)
    b.check({"depth": 3})  # at the limit, OK
    with pytest.raises(RuntimeError, match="depth"):
        b.check({"depth": 4})


def test_budget_total_spawn_limit() -> None:
    b = AgentBudget(max_total_spawns=5)
    b.check({"total_spawns": 5})  # at the limit, OK
    with pytest.raises(RuntimeError, match="spawn count"):
        b.check({"total_spawns": 6})


# ---------------------------------------------------------------------------
# describe_graphs._parse_zoom_decision
# ---------------------------------------------------------------------------


def test_parse_zoom_decision_no_zoom() -> None:
    text = '{"zoom": false}'
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert zoom is False
    assert ranges == []


def test_parse_zoom_decision_fenced_json() -> None:
    text = (
        "Some prose.\n"
        "```json\n"
        '{"zoom": true, "ranges": ['
        f'{{"start": 100, "end": {100 + MIN_ZOOM_SPAN + 5}, "reason": "x"}}'
        "]}\n"
        "```\n"
    )
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert zoom is True
    assert len(ranges) == 1
    assert ranges[0]["start"] == 100


def test_parse_zoom_decision_drops_short_span() -> None:
    short = MIN_ZOOM_SPAN - 1
    text = (
        '{"zoom": true, "ranges": [{'
        f'"start": 10, "end": {10 + short}, "reason": "x"'
        "}]}"
    )
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert ranges == []
    assert zoom is False    # no valid ranges → effectively no zoom


def test_parse_zoom_decision_drops_outside_parent_range() -> None:
    text = (
        '{"zoom": true, "ranges": [{'
        f'"start": 5000, "end": 5100, "reason": "out of range"'
        "}]}"
    )
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert ranges == []
    assert zoom is False


def test_parse_zoom_decision_caps_count() -> None:
    text = (
        '{"zoom": true, "ranges": ['
        f'{{"start": 0, "end": {MIN_ZOOM_SPAN}, "reason": "a"}},'
        f'{{"start": 100, "end": {100 + MIN_ZOOM_SPAN}, "reason": "b"}},'
        f'{{"start": 200, "end": {200 + MIN_ZOOM_SPAN}, "reason": "c"}}'
        "]}"
    )
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert zoom is True
    assert len(ranges) == 2     # capped at MAX_ZOOM_STEPS


def test_parse_zoom_decision_drops_invalid_types() -> None:
    text = (
        '{"zoom": true, "ranges": [{'
        '"start": "80", "end": 130, "reason": "bad start type"'
        "}]}"
    )
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert ranges == []
    assert zoom is False


def test_parse_zoom_decision_no_json_returns_no_zoom() -> None:
    text = "Just prose, no JSON object here."
    zoom, ranges = _parse_zoom_decision(text, 0, 1000)
    assert zoom is False
    assert ranges == []


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


def test_agent_subclass_warns_when_phase_methods_not_overridden(caplog) -> None:
    """Defining a subclass that forgets a phase emits a warning naming
    every missing method."""
    import logging

    with caplog.at_level(logging.WARNING, logger="app.services.llm.agent_framework"):

        class _PartialAgent(Agent):
            name = "_partial_agent_for_test"

            def executor(self, state, step, registry):
                return {}

            # planner / synthesizer / evaluator deliberately not overridden.

    messages = [r.getMessage() for r in caplog.records]
    assert any("_PartialAgent" in m and "planner" in m for m in messages)
    assert any("_PartialAgent" in m and "synthesizer" in m for m in messages)
    assert any("_PartialAgent" in m and "evaluator" in m for m in messages)


def test_agent_subclass_no_warning_when_all_phases_overridden(caplog) -> None:
    """A subclass that overrides every phase (even with None returns)
    produces no missing-override warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="app.services.llm.agent_framework"):

        class _CompleteAgent(Agent):
            name = "_complete_agent_for_test"

            def planner(self, state):
                return None

            def executor(self, state, step, registry):
                return {}

            def synthesizer(self, state):
                return None

            def evaluator(self, state):
                return None

    missing_msgs = [
        r.getMessage() for r in caplog.records
        if "_CompleteAgent" in r.getMessage() and "did not override" in r.getMessage()
    ]
    assert missing_msgs == []


def test_agent_subclass_warns_when_name_missing(caplog) -> None:
    """Subclass without a `name` class attribute warns at definition time."""
    import logging

    with caplog.at_level(logging.WARNING, logger="app.services.llm.agent_framework"):

        class _NamelessAgent(Agent):
            def planner(self, state):
                return None

            def executor(self, state, step, registry):
                return {}

            def synthesizer(self, state):
                return None

            def evaluator(self, state):
                return None

    messages = [r.getMessage() for r in caplog.records]
    assert any("_NamelessAgent" in m and "name" in m for m in messages)


def test_agent_register_drops_unoverridden_phases() -> None:
    """``Agent.register`` records ``None`` for phase methods the subclass
    didn't override, so the framework strips those nodes out of the graph."""

    class _SkipperAgent(Agent):
        name = "_skipper_agent_for_test"

        def executor(self, state, step, registry):
            return {}

        # planner / synthesizer / evaluator NOT overridden — should be
        # None in the spec.

    spec = _SkipperAgent.register()
    try:
        assert spec.planner_fn is None
        assert spec.synthesizer_fn is None
        assert spec.evaluator_fn is None
        assert spec.executor_fn is not None
        assert spec.name == "_skipper_agent_for_test"
        assert AGENT_SPECS[spec.name] is spec
        assert spec.name in AGENT_REGISTRY
    finally:
        AGENT_SPECS.pop(spec.name, None)
        AGENT_REGISTRY.pop(spec.name, None)


def test_agent_register_keeps_overridden_phases() -> None:
    """``Agent.register`` records the overridden methods (even if they
    return None at runtime) so the framework includes the nodes."""

    class _OverridingAgent(Agent):
        name = "_overriding_agent_for_test"

        def planner(self, state):
            return None     # explicit skip — still counts as overridden

        def executor(self, state, step, registry):
            return {}

        def synthesizer(self, state):
            return None

        def evaluator(self, state):
            return None

    spec = _OverridingAgent.register()
    try:
        assert spec.planner_fn is not None
        assert spec.synthesizer_fn is not None
        assert spec.evaluator_fn is not None
    finally:
        AGENT_SPECS.pop(spec.name, None)
        AGENT_REGISTRY.pop(spec.name, None)


def test_agent_register_raises_when_name_unset() -> None:
    class _UnnamedAgent(Agent):
        def planner(self, state):
            return None

        def executor(self, state, step, registry):
            return {}

        def synthesizer(self, state):
            return None

        def evaluator(self, state):
            return None

    with pytest.raises(RuntimeError, match="no `name`"):
        _UnnamedAgent.register()
