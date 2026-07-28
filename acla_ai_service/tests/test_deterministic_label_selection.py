import json
from pathlib import Path

import numpy as np
import pandas as pd

import app.local_annotation_agent.workflow as workflow
import app.local_annotation_agent.workflow.deterministic_facts as deterministic_facts
from app.local_annotation_agent.workflow import deterministic
from app.local_annotation_agent.workflow.deterministic_engine import (
    FactDefinition,
    FactRegistry,
    InclusiveRange,
    InputDefinition,
    InputRegistry,
    MISSING,
    PredicateEvaluation,
    RequirementBranchEvaluation,
    RequirementEvaluation,
    RequirementInterpreter,
    ResolvedInput,
    validate_requirements,
)
from app.local_annotation_agent.workflow.deterministic_facts import (
    EvaluationContext,
    smooth_telemetry,
)


def _requirement(tags, fact, operator="eq", value=True):
    normalized_tags = dict(tags) if isinstance(tags, dict) else list(tags)
    return {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": normalized_tags},
            "condition": {"fact": fact, "operator": operator, "value": value},
        }]}],
    }


def _comparison_tags(player, expert):
    return {"player": player, "expert": expert}


def _evaluate(requirements, df, start=0, end=None, **context_kwargs):
    if end is None:
        end = len(df) - 1
    context = EvaluationContext.from_dataframe(df, **context_kwargs)
    return deterministic.evaluate_requirements(
        requirements, context, InclusiveRange(start, end),
    )


def _branch(index, start, end, text="fact: True"):
    return RequirementBranchEvaluation(index, [
        PredicateEvaluation(True, text, "fact", InclusiveRange(start, end)),
    ])


def _matched(*branches):
    first = branches[0]
    return RequirementEvaluation(
        True, first.branch, first.passed, [], list(branches),
    )


def test_telemetry_is_smoothed_once_with_centered_three_sample_mean():
    df = pd.DataFrame({"signal": [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]})

    telemetry = smooth_telemetry(df)

    np.testing.assert_allclose(
        telemetry["signal"],
        [0.0, 10.0 / 3.0, 10.0 / 3.0, 11.0 / 3.0, 2.0 / 3.0, 1.0],
    )
    assert df["signal"].tolist() == [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]


def test_time_facts_use_the_single_initial_smoothing_pass(monkeypatch):
    from app.shared import annotation_agent_tools

    calls = []
    original_smooth = deterministic_facts.smooth_telemetry

    def track_smoothing(df):
        calls.append(df)
        return original_smooth(df)

    def reject_pipeline_query(*_args, **_kwargs):
        raise AssertionError("time facts must not run a second smoothing path")

    monkeypatch.setattr(deterministic_facts, "smooth_telemetry", track_smoothing)
    monkeypatch.setattr(
        annotation_agent_tools, "run_pipeline_query", reject_pipeline_query,
    )

    result = _evaluate(
        _requirement(
            ["section_range"], "find_total_time_change",
            operator="eq", value=100.0,
        ),
        pd.DataFrame({"expert_time_difference": [0.0, 100.0, 200.0]}),
    )

    assert result.matched
    assert len(calls) == 1


def test_catalog_requirements_are_valid():
    assert deterministic.validate_catalog() == []


def test_msp15_uses_player_reapplication_boundaries_for_handling_balance():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    assert requirements["MSP15"] == _requirement(
        [
            "player_throttle_reapplication_onset_iloc",
            "player_throttle_reapplication_end_iloc",
        ],
        "find_oversteer_or_understeer_between_ilocs",
    )


def test_msp15_checks_player_balance_across_the_combined_reapplication_range(
    monkeypatch,
):
    landmarks = {
        "player": {"reapplication_onset": 4, "reapplication_end": 7},
        "expert": {"reapplication_onset": 2, "reapplication_end": 9},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "throttle" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP15", deterministic.get_label("MSP15"),
    )

    def evaluate(balances):
        size = 12
        values = np.asarray(balances, dtype=float)
        front = np.maximum(-values, 0.0)
        rear = np.maximum(values, 0.0)
        context = EvaluationContext(pd.DataFrame({
            "Physics_slip_angle_front_left": front,
            "Physics_slip_angle_front_right": front,
            "Physics_slip_angle_rear_left": rear,
            "Physics_slip_angle_rear_right": rear,
        }))
        return deterministic.evaluate_requirements(
            requirements, context, InclusiveRange(0, size - 1),
        )

    oversteer = evaluate([0.0] * 5 + [0.03] * 7)
    understeer = evaluate([0.0] * 5 + [-0.03] * 7)

    assert oversteer.matched
    assert understeer.matched
    assert oversteer.matched_branches[0].evidence_range == InclusiveRange(4, 7)
    assert understeer.matched_branches[0].evidence_range == InclusiveRange(4, 7)
    assert not evaluate([0.0] * 12).matched

    landmarks["player"]["reapplication_end"] = None
    missing_boundary = evaluate([0.0] * 5 + [0.03] * 7)

    assert not missing_boundary.matched
    assert "missing input player_throttle_reapplication_end_iloc" in (
        missing_boundary.failed[0]
    )


def test_msp18_requires_increasing_speed_difference_during_player_release():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]
    predicates = [
        {
            "inputs": {
                "tags": _comparison_tags(
                    "player_brake_release_onset_iloc",
                    "expert_brake_release_onset_iloc",
                ),
            },
            "condition": {},
        },
        _requirement(
            _comparison_tags(
                "player_brake_release_end_iloc",
                "expert_brake_release_end_iloc",
            ),
            "compare_ilocs",
            value="later",
        )["any_of"][0]["all_of"][0],
        _requirement(
            [
                "player_brake_release_onset_iloc",
                "player_brake_release_end_iloc",
            ],
            "find_speed_gap_slope",
            operator="gt",
            value=0,
        )["any_of"][0]["all_of"][0],
    ]

    assert requirements["MSP18"] == {
        "enabled": True,
        "any_of": [{"all_of": predicates}],
    }


def test_msp18_matches_only_when_speed_difference_increases_during_release(
    monkeypatch,
):
    landmarks = {
        "player": {"release_onset": 2, "release_end": 6},
        "expert": {"release_onset": 2, "release_end": 5},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP18", deterministic.get_label("MSP18"),
    )

    def evaluate(speed_differences):
        player_speed = np.full(len(speed_differences), 100.0)
        context = EvaluationContext.from_dataframe(pd.DataFrame({
            "Physics_speed_kmh": player_speed,
            "expert_optimal_speed": player_speed + speed_differences,
        }))
        return deterministic.evaluate_requirements(
            requirements, context, InclusiveRange(0, len(speed_differences) - 1),
        )

    increasing = evaluate(np.array([0, 0, 0, 1, 2, 3, 4, 4, 4], dtype=float))
    flat = evaluate(np.zeros(9, dtype=float))
    decreasing = evaluate(np.array([4, 4, 4, 3, 2, 1, 0, 0, 0], dtype=float))

    assert increasing.matched
    assert increasing.matched_branches[0].evidence_range == InclusiveRange(2, 6)
    assert not flat.matched
    assert not decreasing.matched


def test_msp20_uses_aligned_release_and_player_reapplication_stability():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]
    predicates = [
        _requirement(
            _comparison_tags(
                "player_throttle_release_end_iloc",
                "expert_throttle_release_end_iloc",
            ),
            "compare_ilocs",
            value="aligned",
        )["any_of"][0]["all_of"][0],
        _requirement(
            _comparison_tags(
                "player_throttle_reapplication_onset_iloc",
                "expert_throttle_reapplication_onset_iloc",
            ),
            "compare_ilocs",
            value="earlier",
        )["any_of"][0]["all_of"][0],
        _requirement(
            [
                "player_throttle_reapplication_onset_iloc",
                "player_throttle_reapplication_end_iloc",
            ],
            "find_oversteer_or_understeer_between_ilocs",
        )["any_of"][0]["all_of"][0],
    ]

    assert requirements["MSP20"] == {
        "enabled": True,
        "any_of": [{"all_of": predicates}],
    }


def test_msp21_uses_aligned_onset_later_end_and_close_speed():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]
    predicates = [
        _requirement(
            _comparison_tags(
                "player_throttle_reapplication_onset_iloc",
                "expert_throttle_reapplication_onset_iloc",
            ),
            "compare_ilocs",
            value="aligned",
        )["any_of"][0]["all_of"][0],
        _requirement(
            _comparison_tags(
                "player_throttle_reapplication_end_iloc",
                "expert_throttle_reapplication_end_iloc",
            ),
            "compare_ilocs",
            value="later",
        )["any_of"][0]["all_of"][0],
        _requirement(
            ["expert_throttle_reapplication_onset_iloc"],
            "find_speed_difference_at_iloc",
            operator="between",
            value=[-5, 5],
        )["any_of"][0]["all_of"][0],
    ]

    assert requirements["MSP21"] == {
        "enabled": True,
        "any_of": [{"all_of": predicates}],
    }


def test_msp21_speed_tolerance_is_inclusive_at_expert_onset():
    tag = "expert_throttle_reapplication_onset_iloc"
    inputs = InputRegistry({
        tag: InputDefinition(
            "iloc",
            lambda _context, _scope: ResolvedInput(
                tag, "iloc", 1, InclusiveRange(1, 1),
            ),
        ),
    })
    interpreter = RequirementInterpreter(inputs, deterministic.FACT_REGISTRY)
    requirements = _requirement(
        [tag],
        "find_speed_difference_at_iloc",
        operator="between",
        value=[-5, 5],
    )

    for speed_difference, expected in (
        (-5.0, True),
        (5.0, True),
        (-5.1, False),
        (5.1, False),
    ):
        context = EvaluationContext.from_dataframe(pd.DataFrame({
            "Physics_speed_kmh": [100.0] * 3,
            "expert_optimal_speed": [100.0 + speed_difference] * 3,
        }))

        result = interpreter.evaluate(
            requirements,
            context,
            InclusiveRange(0, 2),
        )

        assert result.matched is expected


def test_sub_label_requirements_do_not_depend_on_labels_section():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    catalog = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )

    assert "labels" not in catalog
    assert deterministic.get_label("RM7") is None
    assert deterministic._requirements_for("RM7", None) == (
        catalog["sub_label_selection_requirements"]["RM7"]
    )


def test_every_catalog_predicate_uses_inputs_and_condition():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    catalogs = [
        json.loads((root / "lap_annotation.json").read_text(encoding="utf-8")),
        json.loads((root / "sub_label_annotation.json").read_text(encoding="utf-8")),
    ]
    requirements = [catalogs[0]["selection_requirements"]]
    requirements.extend([
        catalogs[1]["sub_label_selection_requirements"],
        catalogs[1]["segment_type_selection_requirements"],
    ])

    predicates = [
        predicate
        for group in requirements
        for requirement in group.values()
        for branch in requirement.get("any_of", [])
        for predicate in branch.get("all_of", [])
    ]

    assert predicates
    assert all(set(predicate) == {"inputs", "condition"} for predicate in predicates)
    assert all(predicate["inputs"]["tags"] for predicate in predicates)


def test_registry_is_the_source_of_known_tags_and_facts():
    assert "compare_ilocs" in deterministic.FACT_REGISTRY.names()
    assert "compare_upshift_timing" in deterministic.FACT_REGISTRY.names()
    assert "compare_downshift_timing" in deterministic.FACT_REGISTRY.names()
    assert "find_speed_difference_at_iloc" in deterministic.FACT_REGISTRY.names()
    assert "find_speed_gap_slope" in deterministic.FACT_REGISTRY.names()
    assert "find_player_brake_peak" in deterministic.FACT_REGISTRY.names()
    assert "find_trajectory_position_at_iloc" in deterministic.FACT_REGISTRY.names()
    assert "find_trajectory_split" in deterministic.FACT_REGISTRY.names()
    assert "find_trajectory_splitting_wider" not in deterministic.FACT_REGISTRY.names()
    assert (
        "find_oversteer_or_understeer_between_ilocs"
        in deterministic.FACT_REGISTRY.names()
    )
    assert "player_brake_application_onset_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "player_throttle_reapplication_onset_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "player_throttle_reapplication_end_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "player_throttle_application_onset_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert "brake_comparison_range" in deterministic.INPUT_REGISTRY.names()
    assert "corner_entry_start_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "corner_exit_end_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "segment_start_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "segment_end_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "segment_apex_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "segment_start_to_apex_range" not in deterministic.INPUT_REGISTRY.names()
    assert "expert_upshift_range" in deterministic.INPUT_REGISTRY.names()
    assert "expert_downshift_range" in deterministic.INPUT_REGISTRY.names()
    assert "player_upshift_onset_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert "expert_upshift_end_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert "player_upshift_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert not hasattr(deterministic, "KNOWN_FACTS")
    assert not hasattr(deterministic, "FactSet")


def test_throttle_reapplication_uses_the_positive_trend_after_release():
    df = pd.DataFrame({
        "Physics_gas": [
            1.0, 0.702451, 0.125792, 0.0, 0.0,
            0.0, 0.084559, 0.839216, 1.0,
        ],
        "expert_optimal_throttle": [
            1.0, 0.516871, 0.071691, 0.0, 0.096923,
            0.242775, 0.456995, 0.820557, 0.950046,
        ],
    }, index=range(311, 320))
    context = EvaluationContext.from_dataframe(df)
    scope = InclusiveRange(311, 320)

    player_onset = context.resolve_input(
        "player_throttle_reapplication_onset_iloc", scope,
        deterministic.INPUT_REGISTRY,
    )
    player_end = context.resolve_input(
        "player_throttle_reapplication_end_iloc", scope,
        deterministic.INPUT_REGISTRY,
    )
    expert_onset = context.resolve_input(
        "expert_throttle_reapplication_onset_iloc", scope,
        deterministic.INPUT_REGISTRY,
    )
    expert_end = context.resolve_input(
        "expert_throttle_reapplication_end_iloc", scope,
        deterministic.INPUT_REGISTRY,
    )

    assert player_onset is not None and player_onset.value == 317
    assert player_end is not None and player_end.value == 319
    assert expert_onset is not None and expert_onset.value == 315
    assert expert_end is not None and expert_end.value == 319

    requirements = deterministic._requirements_for(
        "MSP15", deterministic.get_label("MSP15"),
    )
    result = deterministic.evaluate_requirements(requirements, context, scope)

    assert not result.matched


def test_throttle_reapplication_is_missing_without_a_positive_trend():
    df = pd.DataFrame({
        "Physics_gas": [1.0, 0.8, 0.6, 0.4],
        "expert_optimal_throttle": [1.0, 0.8, 0.6, 0.4],
    })
    context = EvaluationContext.from_dataframe(df)

    assert context.resolve_input(
        "player_throttle_reapplication_onset_iloc",
        InclusiveRange(0, 3),
        deterministic.INPUT_REGISTRY,
    ) is None


def test_registry_rejects_duplicate_strategy_names():
    definition = FactDefinition(("range",), lambda *_args: True)

    try:
        FactRegistry([("duplicate", definition), ("duplicate", definition)])
    except ValueError as error:
        assert "duplicate fact strategy" in str(error)
    else:
        raise AssertionError("duplicate fact registration was accepted")


def test_interpreter_retains_every_matching_branch_and_its_range():
    definitions = {
        "first": InputDefinition(
            "range", lambda _context, _scope: ResolvedInput(
                "first", "range", InclusiveRange(1, 3), InclusiveRange(1, 3),
            ),
        ),
        "second": InputDefinition(
            "range", lambda _context, _scope: ResolvedInput(
                "second", "range", InclusiveRange(7, 9), InclusiveRange(7, 9),
            ),
        ),
    }
    interpreter = RequirementInterpreter(
        InputRegistry(definitions),
        FactRegistry({"present": FactDefinition(("range",), lambda *_args: True)}),
    )
    requirements = {
        "enabled": True,
        "any_of": [
            _requirement(["first"], "present")["any_of"][0],
            _requirement(["second"], "present")["any_of"][0],
        ],
    }
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(11)))

    result = interpreter.evaluate(requirements, context, InclusiveRange(0, 10))

    assert result.matched and result.branch == 0
    assert [branch.evidence_range for branch in result.matched_branches] == [
        InclusiveRange(1, 3), InclusiveRange(7, 9),
    ]


def test_branch_evidence_is_the_envelope_of_all_predicate_inputs():
    ranges = {"first": InclusiveRange(2, 4), "second": InclusiveRange(8, 9)}
    inputs = InputRegistry({
        tag: InputDefinition(
            "range",
            lambda _context, _scope, tag=tag: ResolvedInput(
                tag, "range", ranges[tag], ranges[tag],
            ),
        )
        for tag in ranges
    })
    facts = FactRegistry({
        "present": FactDefinition(("range",), lambda *_args: True),
    })
    requirements = {
        "enabled": True,
        "any_of": [{"all_of": [
            _requirement(["first"], "present")["any_of"][0]["all_of"][0],
            _requirement(["second"], "present")["any_of"][0]["all_of"][0],
        ]}],
    }
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(11)))

    result = RequirementInterpreter(inputs, facts).evaluate(
        requirements, context, InclusiveRange(0, 10),
    )

    assert result.matched_branches[0].evidence_range == InclusiveRange(2, 9)


def test_array_tags_form_one_condition_input_from_their_envelope():
    resolved = {
        "first": ResolvedInput(
            "first", "iloc", 8, InclusiveRange(8, 8),
        ),
        "middle_range": ResolvedInput(
            "middle_range", "range",
            InclusiveRange(2, 4), InclusiveRange(2, 4),
        ),
        "last": ResolvedInput(
            "last", "iloc", 10, InclusiveRange(10, 10),
        ),
    }
    inputs = InputRegistry({
        tag: InputDefinition(
            value.kind,
            lambda _context, _scope, value=value: value,
        )
        for tag, value in resolved.items()
    })
    received = []

    def calculate(_context, values):
        received.append(values)
        return True

    facts = FactRegistry({
        "point_fact": FactDefinition(("iloc",), calculate),
        "range_fact": FactDefinition(("range",), calculate),
    })
    interpreter = RequirementInterpreter(inputs, facts)
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(12)))

    cases = (
        (["first"], "point_fact", InclusiveRange(8, 8), "iloc"),
        (["middle_range"], "range_fact", InclusiveRange(2, 4), "range"),
        (["first", "last"], "range_fact", InclusiveRange(8, 10), "range"),
        (
            ["first", "middle_range", "last"],
            "range_fact",
            InclusiveRange(2, 10),
            "range",
        ),
    )
    for tags, fact, expected_range, expected_kind in cases:
        result = interpreter.evaluate(
            _requirement(tags, fact),
            context,
            InclusiveRange(0, 11),
        )

        assert result.matched
        assert result.matched_branches[0].evidence_range == expected_range
        assert len(received[-1]) == 1
        assert received[-1][0].kind == expected_kind
        assert received[-1][0].evidence_range == expected_range


def test_evidence_only_predicate_contributes_range_without_a_fact():
    ranges = {
        "first": InclusiveRange(2, 2),
        "second": InclusiveRange(9, 9),
    }
    inputs = InputRegistry({
        tag: InputDefinition(
            "iloc",
            lambda _context, _scope, tag=tag: ResolvedInput(
                tag, "iloc", ranges[tag].start, ranges[tag],
            ),
        )
        for tag in ranges
    })
    requirements = {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": ["first", "second"]},
            "condition": {},
        }]}],
    }
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(11)))

    result = RequirementInterpreter(inputs, FactRegistry({})).evaluate(
        requirements, context, InclusiveRange(0, 10),
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(2, 9)
    assert result.passed == ["evidence: inputs resolved"]


def test_evidence_only_predicate_fails_closed_when_an_input_is_missing():
    inputs = InputRegistry({
        "present": InputDefinition(
            "iloc", lambda _context, _scope: ResolvedInput(
                "present", "iloc", 2, InclusiveRange(2, 2),
            ),
        ),
        "missing": InputDefinition("iloc", lambda *_args: None),
    })
    requirements = {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": ["present", "missing"]},
            "condition": {},
        }]}],
    }
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(5)))

    result = RequirementInterpreter(inputs, FactRegistry({})).evaluate(
        requirements, context, InclusiveRange(0, 4),
    )

    assert not result.matched
    assert result.failed == ["evidence: unavailable (missing input missing)"]


def test_missing_input_fails_closed_with_rejected_reason():
    result = _evaluate(
        _requirement(
            _comparison_tags(
                "player_brake_release_end_iloc",
                "expert_brake_release_end_iloc",
            ),
            "compare_ilocs", value="later",
        ),
        pd.DataFrame({
            "Physics_brake": [np.nan] * 5,
            "expert_optimal_brake": [np.nan] * 5,
        }),
    )

    assert not result.matched
    assert "missing input" in result.failed[0]


def test_input_and_fact_strategies_are_cached_per_scope():
    calls = {"input": 0, "fact": 0}

    def resolve(_context, scope):
        calls["input"] += 1
        return ResolvedInput("scope", "range", scope, scope)

    def calculate(_context, _inputs):
        calls["fact"] += 1
        return True

    interpreter = RequirementInterpreter(
        InputRegistry({"scope": InputDefinition("range", resolve)}),
        FactRegistry({"present": FactDefinition(("range",), calculate)}),
    )
    predicate = _requirement(["scope"], "present")["any_of"][0]
    requirements = {"enabled": True, "any_of": [predicate, predicate]}
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(5)))

    interpreter.evaluate(requirements, context, InclusiveRange(0, 4))

    assert calls == {"input": 1, "fact": 1}


def test_validator_rejects_fact_input_kind_mismatch():
    requirements = _requirement(["point"], "range_fact")
    errors = validate_requirements(
        requirements,
        InputRegistry({"point": InputDefinition("iloc", lambda *_args: None)}),
        FactRegistry({"range_fact": FactDefinition(("range",), lambda *_args: True)}),
    )

    assert errors == [
        "branch 0 predicate 0: 'range_fact' expects ('range',), got ('iloc',)"
    ]


def test_validator_derives_range_kind_and_accepts_point_comparison_object():
    inputs = InputRegistry({
        "player_point": InputDefinition("iloc", lambda *_args: None),
        "expert_point": InputDefinition("iloc", lambda *_args: None),
        "start_point": InputDefinition("iloc", lambda *_args: None),
        "end_point": InputDefinition("iloc", lambda *_args: None),
    })
    facts = FactRegistry({
        "range_fact": FactDefinition(("range",), lambda *_args: True),
        "point_comparison": FactDefinition(
            ("iloc", "iloc"), lambda *_args: True,
        ),
    })

    assert validate_requirements(
        _requirement(["player_point", "expert_point"], "range_fact"),
        inputs,
        facts,
    ) == []
    assert validate_requirements(
        _requirement(
            _comparison_tags("player_point", "expert_point"),
            "point_comparison",
        ),
        inputs,
        facts,
    ) == []
    assert validate_requirements(
        _requirement(
            {"start": "start_point", "end": "end_point"},
            "point_comparison",
        ),
        inputs,
        facts,
    ) == []


def test_start_end_object_passes_points_in_semantic_order_and_envelopes_evidence():
    received = []
    inputs = InputRegistry({
        "start_point": InputDefinition(
            "iloc", lambda _context, _scope: ResolvedInput(
                "start_point", "iloc", 2, InclusiveRange(2, 2),
            ),
        ),
        "end_point": InputDefinition(
            "iloc", lambda _context, _scope: ResolvedInput(
                "end_point", "iloc", 7, InclusiveRange(7, 7),
            ),
        ),
    })
    facts = FactRegistry({
        "point_pair": FactDefinition(
            ("iloc", "iloc"),
            lambda _context, values: received.extend(values) or True,
        ),
    })
    requirements = _requirement(
        {"end": "end_point", "start": "start_point"},
        "point_pair",
    )

    result = RequirementInterpreter(inputs, facts).evaluate(
        requirements,
        EvaluationContext.from_dataframe(pd.DataFrame(index=range(9))),
        InclusiveRange(0, 8),
    )

    assert result.matched
    assert [value.tag for value in received] == ["start_point", "end_point"]
    assert result.matched_branches[0].evidence_range == InclusiveRange(2, 7)


def test_validator_rejects_malformed_or_unknown_point_objects():
    inputs = InputRegistry({
        "player_point": InputDefinition("iloc", lambda *_args: None),
        "expert_point": InputDefinition("iloc", lambda *_args: None),
        "start_point": InputDefinition("iloc", lambda *_args: None),
        "end_point": InputDefinition("iloc", lambda *_args: None),
    })
    facts = FactRegistry({
        "point_comparison": FactDefinition(
            ("iloc", "iloc"), lambda *_args: True,
        ),
    })

    for tags in (
        {},
        {"player": "player_point"},
        {"player": "player_point", "expert": 1},
        {"player": "player_point", "expert": "expert_point", "extra": "point"},
        {"start": "start_point"},
        {"start": "start_point", "end": 1},
        {"start": "start_point", "end": "end_point", "extra": "point"},
    ):
        requirements = {
            "enabled": True,
            "any_of": [{"all_of": [{
                "inputs": {"tags": tags},
                "condition": {
                    "fact": "point_comparison",
                    "operator": "eq",
                    "value": True,
                },
            }]}],
        }

        assert validate_requirements(requirements, inputs, facts) == [
            "branch 0 predicate 0: tags must be a non-empty string list "
            "or a player/expert or start/end string object"
        ]

    unknown = _requirement(
        _comparison_tags("player_point", "missing"),
        "point_comparison",
    )
    assert validate_requirements(unknown, inputs, facts) == [
        "branch 0 predicate 0: unknown input tag 'missing'",
        "branch 0 predicate 0: 'point_comparison' expects ('iloc', 'iloc'), "
        "got ('iloc',)",
    ]


def test_validator_accepts_empty_condition_and_rejects_partial_condition():
    inputs = InputRegistry({
        "point": InputDefinition("iloc", lambda *_args: None),
    })
    facts = FactRegistry({
        "present": FactDefinition(("iloc",), lambda *_args: True),
    })
    evidence_only = {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": ["point"]},
            "condition": {},
        }]}],
    }
    partial = {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": ["point"]},
            "condition": {"fact": "present"},
        }]}],
    }

    assert validate_requirements(evidence_only, inputs, facts) == []
    assert validate_requirements(partial, inputs, facts) == [
        "branch 0 predicate 0: condition must be empty or contain fact, operator, and value"
    ]


def test_compare_ilocs_uses_explicit_roles_and_their_point_envelope():
    df = pd.DataFrame({
        "Physics_brake": [0, 0, 0, 0, 1, 1, 0],
        "expert_optimal_brake": [0, 1, 1, 0, 0, 0, 0],
    })
    result = _evaluate(
        _requirement(
            {
                "expert": "expert_brake_application_onset_iloc",
                "player": "player_brake_application_onset_iloc",
            },
            "compare_ilocs", value="later",
        ),
        df,
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(0, 3)


def test_compare_ilocs_requires_exact_alignment():
    compare_ilocs = deterministic.FACT_REGISTRY.get("compare_ilocs")
    assert compare_ilocs is not None

    for player, expert, expected in (
        (5, 5, "aligned"),
        (4, 5, "earlier"),
        (6, 5, "later"),
    ):
        inputs = [
            ResolvedInput(
                "player", "iloc", player, InclusiveRange(player, player),
            ),
            ResolvedInput(
                "expert", "iloc", expert, InclusiveRange(expert, expert),
            ),
        ]
        assert compare_ilocs.calculate(None, inputs) == expected


def test_steering_landmarks_use_exact_iloc_comparison(monkeypatch):
    landmarks = {
        "player": {"apex": 6},
        "expert": {"apex": 5},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_steering_landmarks",
        lambda _context, _scope, driver: landmarks[driver],
    )
    requirements = deterministic._requirements_for(
        "MSP3", deterministic.get_label("MSP3"),
    )

    result = _evaluate(
        requirements, pd.DataFrame(index=range(10)), end=9,
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(5, 6)


def test_expert_shift_ranges_bracket_first_contiguous_directional_change():
    df = pd.DataFrame({
        "expert_optimal_gear": [2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 3, 2, 2, 2],
    })
    context = EvaluationContext.from_dataframe(df)
    scope = InclusiveRange(0, 13)

    for tag, expected in (
        ("expert_upshift_range", InclusiveRange(1, 7)),
        ("expert_downshift_range", InclusiveRange(7, 12)),
    ):
        resolved = context.resolve_input(tag, scope, deterministic.INPUT_REGISTRY)
        assert resolved is not None
        assert resolved.value == expected
        assert resolved.evidence_range == expected


def test_expert_shift_range_is_unavailable_without_requested_direction():
    context = EvaluationContext.from_dataframe(pd.DataFrame({
        "expert_optimal_gear": [4, 4, 4, 3, 3, 3],
    }))

    resolved = context.resolve_input(
        "expert_upshift_range", InclusiveRange(0, 5),
        deterministic.INPUT_REGISTRY,
    )

    assert resolved is None


def test_shift_timing_compares_boundary_progress_inside_expert_range():
    cases = [
        ("MSP35", [4, 4, 4, 4, 4, 4], [2, 2, 2, 3, 3, 3]),
        ("MSP36", [1, 1, 1, 1, 1, 1], [2, 2, 2, 3, 3, 3]),
        ("MSP37", [1, 1, 1, 1, 1, 1], [3, 3, 3, 2, 2, 2]),
        ("MSP38", [4, 4, 4, 4, 4, 4], [3, 3, 3, 2, 2, 2]),
    ]

    for label_id, player, expert in cases:
        requirements = deterministic._requirements_for(
            label_id, deterministic.get_label(label_id),
        )
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_gear": player,
            "expert_optimal_gear": expert,
        }))

        assert result.matched, label_id
        assert result.matched_branches[0].evidence_range == InclusiveRange(1, 4)


def test_shift_timing_tolerates_one_gear_early_or_late():
    cases = [
        ("MSP35", [3, 3, 3, 3, 3, 3], [2, 2, 2, 3, 3, 3]),
        ("MSP36", [2, 2, 2, 2, 2, 2], [2, 2, 2, 3, 3, 3]),
        ("MSP37", [2, 2, 2, 2, 2, 2], [3, 3, 3, 2, 2, 2]),
        ("MSP38", [3, 3, 3, 3, 3, 3], [3, 3, 3, 2, 2, 2]),
    ]

    for label_id, player, expert in cases:
        requirements = deterministic._requirements_for(
            label_id, deterministic.get_label(label_id),
        )
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_gear": player,
            "expert_optimal_gear": expert,
        }))

        assert not result.matched, label_id


def test_shift_timing_aligns_matching_changes_and_rejects_ambiguous_changes():
    expert = [2, 2, 2, 3, 3, 3]
    early_requirements = deterministic._requirements_for(
        "MSP35", deterministic.get_label("MSP35"),
    )
    late_requirements = deterministic._requirements_for(
        "MSP36", deterministic.get_label("MSP36"),
    )
    aligned_requirements = _requirement(
        ["expert_upshift_range"], "compare_upshift_timing",
        operator="between", value=[-1, 1],
    )

    aligned = pd.DataFrame({
        "Physics_gear": expert,
        "expert_optimal_gear": expert,
    })
    assert not _evaluate(early_requirements, aligned).matched
    assert not _evaluate(late_requirements, aligned).matched
    assert _evaluate(aligned_requirements, aligned).matched

    for player in (
        [3, 3, 3, 2, 2, 2],
        [1, 1, 1, 3, 3, 3],
        [2, 2, 2, 4, 4, 4],
    ):
        ambiguous = pd.DataFrame({
            "Physics_gear": player,
            "expert_optimal_gear": expert,
        })
        assert not _evaluate(early_requirements, ambiguous).matched
        assert not _evaluate(late_requirements, ambiguous).matched


def test_shift_timing_uses_expert_event_range_for_reported_regression():
    index = range(33, 51)
    df = pd.DataFrame({
        "Physics_gear": [5, 5, 5, 5, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "expert_optimal_gear": [5, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    }, index=index)
    context = EvaluationContext.from_dataframe(df)
    evaluated = deterministic.evaluate_labels(
        ["MSP35", "MSP36", "MSP37", "MSP38"],
        context, InclusiveRange(33, 50),
    )

    assert evaluated.labels == ["MSP38"]
    branch = evaluated.evaluations["MSP38"].matched_branches[0]
    assert branch.evidence_range == InclusiveRange(34, 39)

    result = deterministic.calculate_detailed_annotation(
        df,
        parent_start=33,
        parent_end=50,
        parent_main_labels=["MSP"],
    )
    annotation = next(
        value for value in result.label_annotations
        if value["label_id"] == "MSP38"
    )
    assert (annotation["start_index"], annotation["end_index"]) == (34, 39)
    assert "MSP35" not in result.final_labels


def test_brake_comparison_range_uses_both_complete_braking_periods(monkeypatch):
    landmarks = {
        "player": {"application_onset": 4, "release_end": 9},
        "expert": {"application_onset": 2, "release_end": 7},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(12)))

    resolved = context.resolve_input(
        "brake_comparison_range",
        InclusiveRange(0, 11),
        deterministic.INPUT_REGISTRY,
    )

    assert resolved is not None
    assert resolved.value == InclusiveRange(2, 9)
    assert resolved.evidence_range == InclusiveRange(2, 9)


def test_msp54_and_msp55_use_the_exact_four_brake_landmark_tags():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]
    tags = [
        "player_brake_application_onset_iloc",
        "expert_brake_application_onset_iloc",
        "player_brake_release_end_iloc",
        "expert_brake_release_end_iloc",
    ]

    assert requirements["MSP54"] == _requirement(
        tags, "compare_gear_range", value="lower",
    )
    assert requirements["MSP55"] == _requirement(
        tags, "compare_gear_range", value="higher",
    )


def test_compare_gear_range_classifies_comparable_samples():
    fact = deterministic.FACT_REGISTRY.get("compare_gear_range")
    assert fact is not None
    range_ = InclusiveRange(0, 2)
    inputs = [ResolvedInput("combined_range", "range", range_, range_)]

    for player, expert, expected in (
        ([2, 2, 3], [3, 3, 4], "lower"),
        ([4, 4, 3], [3, 3, 2], "higher"),
        ([3, 2, 4], [3, 2, 4], "aligned"),
        ([2, 4, 3], [3, 3, 3], "mixed"),
        ([2, np.nan, np.nan], [np.nan, 3, 3], MISSING),
    ):
        context = EvaluationContext(pd.DataFrame({
            "Physics_gear": player,
            "expert_optimal_gear": expert,
        }))
        actual = fact.calculate(context, inputs)

        if expected is MISSING:
            assert actual is MISSING
        else:
            assert actual == expected


def test_msp54_and_msp55_select_by_four_landmark_gear_envelope(monkeypatch):
    landmarks = {
        "player": {"application_onset": 4, "release_end": 9},
        "expert": {"application_onset": 2, "release_end": 7},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    expert = [3] * 12

    def evaluate(player):
        context = EvaluationContext(pd.DataFrame({
            "Physics_gear": player,
            "expert_optimal_gear": expert,
        }))
        return deterministic.evaluate_labels(
            ["MSP54", "MSP55"], context, InclusiveRange(0, 11),
        )

    lower = evaluate([2] * 12)
    higher = evaluate([4] * 12)
    aligned = evaluate([3] * 12)
    mixed = evaluate([2] * 6 + [4] * 6)

    assert lower.labels == ["MSP54"]
    assert higher.labels == ["MSP55"]
    assert aligned.labels == []
    assert mixed.labels == []
    assert lower.evaluations["MSP54"].matched_branches[
        0
    ].evidence_range == InclusiveRange(2, 9)


def test_msp22_checks_speed_difference_at_brake_application_end(monkeypatch):
    landmarks = {
        "player": {
            "application_onset": 2,
            "application_end": 6,
            "release_end": 7,
            "peak": 0.5,
        },
        "expert": {
            "application_onset": 2,
            "application_end": 5,
            "peak": 0.8,
        },
    }

    def fake_landmarks(_context, _scope, driver, control):
        assert control == "brake"
        return landmarks[driver]

    monkeypatch.setattr(deterministic_facts, "_control_landmarks", fake_landmarks)
    requirements = deterministic._requirements_for(
        "MSP22", deterministic.get_label("MSP22"),
    )
    speed_gap = [4.0, 4.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0]
    df = pd.DataFrame({
        "Physics_brake": [0.0, 0.0, 0.2, 0.4, 0.5, 0.3, 0.2, 0.0, 0.0],
        "expert_optimal_brake": [0.8] * 9,
        "Physics_speed_kmh": [100.0 - gap for gap in speed_gap],
        "expert_optimal_speed": [100.0] * 9,
    })

    matched = _evaluate(requirements, df)

    assert matched.matched
    assert matched.matched_branches[0].evidence_range == InclusiveRange(2, 7)
    predicates = matched.matched_branches[0].predicates
    assert predicates[0].evidence_range == InclusiveRange(6, 6)
    assert predicates[1].evidence_range == InclusiveRange(2, 6)
    assert predicates[2].evidence_range == InclusiveRange(2, 7)

    landmarks["player"]["application_end"] = None
    missing_endpoint = _evaluate(requirements, df)

    assert not missing_endpoint.matched
    assert "missing input" in missing_endpoint.failed[0]


def test_msp22_matches_mean_smoothed_player_brake_peak_regression():
    df = pd.DataFrame({
        "Physics_brake": [
            0.0, 0.0, 0.28, 0.98, 0.55, 0.0, 0.0, 0.46, 0.0, 0.0, 0.0,
        ],
        "expert_optimal_brake": [
            0.0, 0.0, 0.15, 0.64, 0.46, 0.27, 0.01, 0.0, 0.0, 0.0, 0.0,
        ],
        "Physics_speed_kmh": [
            100.0, 100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0, 76.0, 73.0,
        ],
        "expert_optimal_speed": [100.0] * 11,
    }, index=range(53, 64))
    context = EvaluationContext.from_dataframe(df)

    evaluated = deterministic.evaluate_labels(
        ["MSP13", "MSP22"], context, InclusiveRange(53, 63),
    )

    assert evaluated.labels == ["MSP22"]
    branch = evaluated.evaluations["MSP22"].matched_branches[0]
    assert branch.evidence_range == InclusiveRange(54, 62)


def test_msp22_brake_window_matches_growing_speed_gap(monkeypatch):
    landmarks = {
        "player": {
            "application_onset": 2,
            "application_end": 6,
            "release_end": 7,
            "peak": 0.5,
        },
        "expert": {
            "application_onset": 2,
            "application_end": 5,
            "release_end": 7,
            "peak": 0.8,
        },
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP22", deterministic.get_label("MSP22"),
    )
    active_brake = [0.0, 0.0, 0.2, 0.4, 0.5, 0.3, 0.2, 0.0, 0.0]

    for speed_gap in (
        [6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0],
        [6.0, 6.0, 6.0, 10.0, 8.0, 11.0, 12.0, 12.0, 12.0],
        [4.0, 4.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0],
        [100.0, 6.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0],
    ):
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_brake": active_brake,
            "expert_optimal_brake": [0.8] * 9,
            "Physics_speed_kmh": [100.0 - gap for gap in speed_gap],
            "expert_optimal_speed": [100.0] * 9,
        }))

        assert result.matched
        assert result.matched_branches[0].branch == 0
        assert result.matched_branches[0].evidence_range == InclusiveRange(2, 7)


def test_msp22_brake_window_requires_each_condition(monkeypatch):
    landmarks = {
        "player": {
            "application_onset": 2,
            "application_end": 6,
            "release_end": 7,
            "peak": 0.5,
        },
        "expert": {
            "application_onset": 2,
            "application_end": 5,
            "release_end": 7,
            "peak": 0.8,
        },
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP22", deterministic.get_label("MSP22"),
    )
    active_brake = [0.0, 0.0, 0.2, 0.4, 0.5, 0.3, 0.2, 0.0, 0.0]
    growing_gap = [6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0]
    cases = (
        ([6.0] * 9, active_brake),
        ([10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 6.0, 6.0], active_brake),
        (growing_gap, [0.9, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
    )

    for speed_gap, player_brake in cases:
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_brake": player_brake,
            "expert_optimal_brake": [0.8] * 9,
            "Physics_speed_kmh": [100.0 - gap for gap in speed_gap],
            "expert_optimal_speed": [100.0] * 9,
        }))

        assert not result.matched


def test_msp22_brake_window_fails_closed_for_missing_or_invalid_inputs(monkeypatch):
    landmarks = {
        "player": {
            "application_onset": 2,
            "application_end": 6,
            "release_end": 7,
            "peak": 0.5,
        },
        "expert": {
            "application_onset": 2,
            "application_end": 5,
            "release_end": 7,
            "peak": 0.8,
        },
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP22", deterministic.get_label("MSP22"),
    )
    complete = {
        "Physics_brake": [0.0, 0.0, 0.2, 0.4, 0.5, 0.3, 0.2, 0.0, 0.0],
        "expert_optimal_brake": [0.8] * 9,
        "Physics_speed_kmh": [94.0, 94.0, 94.0, 93.0, 92.0, 91.0, 90.0, 90.0, 90.0],
        "expert_optimal_speed": [100.0] * 9,
    }

    for missing_column in ("Physics_brake", "Physics_speed_kmh"):
        incomplete = {
            name: values for name, values in complete.items()
            if name != missing_column
        }
        assert not _evaluate(requirements, pd.DataFrame(incomplete)).matched

    landmarks["player"]["application_end"] = None
    assert not _evaluate(requirements, pd.DataFrame(complete)).matched

    landmarks["player"]["release_end"] = 2
    assert not _evaluate(requirements, pd.DataFrame(complete)).matched

    landmarks["player"]["release_end"] = 1
    assert not _evaluate(requirements, pd.DataFrame(complete)).matched


def test_msp13_requires_similar_or_higher_speed_at_corner_entry_start(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 7}]},
    )
    requirements = deterministic._requirements_for(
        "MSP13", deterministic.get_label("MSP13"),
    )

    for player_speed, expected in (
        (105.0, True),
        (100.0, True),
        (95.0, True),
        (94.9, False),
    ):
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_brake": [0.4] * 8,
            "expert_optimal_brake": [0.8] * 8,
            "Physics_speed_kmh": [player_speed] * 8,
            "expert_optimal_speed": [100.0] * 8,
        }), end=7)

        assert result.matched is expected


def test_msp13_speed_check_uses_entry_start_not_later_entry_speed(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 7}]},
    )
    requirements = deterministic._requirements_for(
        "MSP13", deterministic.get_label("MSP13"),
    )
    result = _evaluate(requirements, pd.DataFrame({
        "Physics_brake": [0.4] * 8,
        "expert_optimal_brake": [0.8] * 8,
        "Physics_speed_kmh": [80.0, 100.0, 100.0, 100.0, 80.0, 80.0, 80.0, 80.0],
        "expert_optimal_speed": [100.0] * 8,
    }), end=7)

    assert result.matched


def test_msp13_fails_closed_without_entry_geometry_or_speed(monkeypatch):
    requirements = deterministic._requirements_for(
        "MSP13", deterministic.get_label("MSP13"),
    )
    low_brake_only = pd.DataFrame({
        "Physics_brake": [0.4] * 8,
        "expert_optimal_brake": [0.8] * 8,
    })

    assert not _evaluate(requirements, low_brake_only, end=7).matched

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 7}]},
    )

    assert not _evaluate(requirements, low_brake_only, end=7).matched


def test_msp1_requires_player_speed_gap_below_ten_at_expert_brake_onset(
    monkeypatch,
):
    landmarks = {
        "player": {"application_onset": 4, "application_end": 8},
        "expert": {"application_onset": 2, "application_end": 6},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP1", deterministic.get_label("MSP1"),
    )

    for speed_gap, expected in ((9.9, True), (10.0, False), (10.1, False)):
        result = _evaluate(requirements, pd.DataFrame({
            "Physics_speed_kmh": [100.0 - speed_gap] * 12,
            "expert_optimal_speed": [100.0] * 12,
        }), end=11)

        assert result.matched is expected
        if expected:
            assert result.matched_branches[0].evidence_range == InclusiveRange(2, 8)


def test_msp1_application_ends_only_supply_range_and_remain_required(monkeypatch):
    landmarks = {
        "player": {"application_onset": 7, "application_end": 8},
        "expert": {"application_onset": 2, "application_end": 10},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP1", deterministic.get_label("MSP1"),
    )
    df = pd.DataFrame({
        "Physics_speed_kmh": [95.0] * 5 + [80.0] * 7,
        "expert_optimal_speed": [100.0] * 12,
    })

    expert_end_later = _evaluate(requirements, df, end=11)
    assert expert_end_later.matched
    assert expert_end_later.matched_branches[0].evidence_range == InclusiveRange(2, 10)

    landmarks["player"]["application_end"] = 10
    landmarks["expert"]["application_end"] = 8
    player_end_later = _evaluate(requirements, df, end=11)
    assert player_end_later.matched
    assert player_end_later.matched_branches[0].evidence_range == InclusiveRange(2, 10)

    landmarks["player"]["application_end"] = None
    missing_end = _evaluate(requirements, df, end=11)
    assert not missing_end.matched
    assert "evidence: unavailable (missing input" in missing_end.failed[0]


def test_early_application_label_requires_matching_onset_and_end(monkeypatch):
    landmarks = {
        "player": {"application_onset": 1, "application_end": 10},
        "expert": {"application_onset": 5, "application_end": 8},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP5", deterministic.get_label("MSP5"),
    )

    conflicting_end = _evaluate(
        requirements, pd.DataFrame(index=range(12)), end=11,
    )
    assert not conflicting_end.matched

    landmarks["player"]["application_end"] = 4
    matched = _evaluate(requirements, pd.DataFrame(index=range(12)), end=11)
    assert matched.matched
    assert matched.matched_branches[0].evidence_range == InclusiveRange(1, 8)

    landmarks["player"]["application_end"] = None
    missing_end = _evaluate(requirements, pd.DataFrame(index=range(12)), end=11)
    assert not missing_end.matched
    assert "missing input" in missing_end.failed[0]


def test_release_initiation_labels_compare_matching_control_endpoints(monkeypatch):
    landmarks = {
        ("player", "brake"): {
            "application_onset": 4,
            "release_onset": 8,
            "release_end": 12,
        },
        ("expert", "brake"): {
            "application_onset": 4,
            "release_onset": 4,
            "release_end": 8,
        },
        ("player", "throttle"): {"release_onset": 9, "release_end": 13},
        ("expert", "throttle"): {"release_onset": 5, "release_end": 9},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: landmarks[(driver, control)],
    )

    for label_id, expected_range in (
        ("MSP27", InclusiveRange(4, 12)),
        ("MSP29", InclusiveRange(5, 13)),
    ):
        requirements = deterministic._requirements_for(
            label_id, deterministic.get_label(label_id),
        )
        result = _evaluate(
            requirements, pd.DataFrame(index=range(15)), end=14,
        )
        assert result.matched
        assert result.matched_branches[0].evidence_range == expected_range

    landmarks[("player", "brake")]["application_onset"] = 8
    requirements = deterministic._requirements_for(
        "MSP27", deterministic.get_label("MSP27"),
    )
    mismatched_onset = _evaluate(
        requirements, pd.DataFrame(index=range(15)), end=14,
    )
    assert not mismatched_onset.matched


def test_msp23_rejects_release_onsets_one_iloc_apart(monkeypatch):
    landmarks = {
        "player": {"release_onset": 35, "release_end": 38},
        "expert": {"release_onset": 36, "release_end": 41},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "throttle" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP23", deterministic.get_label("MSP23"),
    )
    df = pd.DataFrame(index=range(33, 52))

    misaligned = _evaluate(requirements, df, start=33, end=51)

    assert not misaligned.matched
    assert "compare_ilocs: 'earlier'" in misaligned.failed

    landmarks["expert"]["release_onset"] = 35
    aligned = _evaluate(requirements, df, start=33, end=51)

    assert aligned.matched
    assert aligned.matched_branches[0].evidence_range == InclusiveRange(35, 41)


def test_brake_hold_lengths_one_iloc_apart_are_not_aligned(monkeypatch):
    landmarks = {
        "player": {"hold_length": 5},
        "expert": {"hold_length": 6},
    }
    monkeypatch.setattr(
        deterministic_facts,
        "_control_landmarks",
        lambda _context, _scope, driver, control: (
            landmarks[driver] if control == "brake" else {}
        ),
    )
    requirements = deterministic._requirements_for(
        "MSP31", deterministic.get_label("MSP31"),
    )

    result = _evaluate(
        requirements, pd.DataFrame(index=range(10)), end=9,
    )

    assert result.matched


def test_every_control_onset_comparison_has_matching_end_evidence():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    for label_id, requirement in requirements.items():
        if not requirement.get("enabled", True):
            continue
        for branch in requirement.get("any_of", []):
            predicates = branch.get("all_of", [])
            by_tags = {
                tuple(
                    predicate["inputs"]["tags"].values()
                    if isinstance(predicate["inputs"]["tags"], dict)
                    else predicate["inputs"]["tags"]
                ): predicate["condition"]
                for predicate in predicates
            }
            for control, phases in (
                ("brake", ("application", "release")),
                ("throttle", ("reapplication", "release")),
            ):
                for phase in phases:
                    onset_tags = (
                        f"player_{control}_{phase}_onset_iloc",
                        f"expert_{control}_{phase}_onset_iloc",
                    )
                    if onset_tags not in by_tags:
                        continue
                    end_tags = (
                        f"player_{control}_{phase}_end_iloc",
                        f"expert_{control}_{phase}_end_iloc",
                    )
                    if (
                        label_id == "MSP20"
                        and control == "throttle"
                        and phase == "reapplication"
                    ):
                        player_interval_tags = (
                            "player_throttle_reapplication_onset_iloc",
                            "player_throttle_reapplication_end_iloc",
                        )
                        assert by_tags[player_interval_tags] == {
                            "fact": "find_oversteer_or_understeer_between_ilocs",
                            "operator": "eq",
                            "value": True,
                        }
                        continue
                    assert end_tags in by_tags, f"{label_id} omits {phase}_end"
                    onset = by_tags[onset_tags]
                    if onset["operator"] == "exists" or onset["value"] in {
                        "earlier", "later",
                    }:
                        if label_id == "MSP1":
                            assert by_tags[end_tags] == {}
                        else:
                            assert by_tags[end_tags] == onset


def test_every_shift_timing_requirement_uses_its_expert_range():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    for label_id, direction, operator, value in (
        ("MSP35", "up", "gt", 1),
        ("MSP36", "up", "lt", -1),
        ("MSP37", "down", "gt", 1),
        ("MSP38", "down", "lt", -1),
    ):
        assert requirements[label_id] == _requirement(
            [f"expert_{direction}shift_range"],
            f"compare_{direction}shift_timing",
            operator=operator,
            value=value,
        )


def test_range_fact_inspects_only_its_declared_range():
    df = pd.DataFrame({
        "Physics_speed_kmh": [0, 100, 100, 100, 100, 0],
        "expert_optimal_speed": [100, 90, 90, 90, 90, 100],
    })
    context = EvaluationContext.from_dataframe(df)
    requirements = _requirement(
        ["speed_comparison_range"], "find_speed_expert_faster", value=False,
    )

    result = deterministic.evaluate_requirements(
        requirements, context, InclusiveRange(2, 3),
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(2, 3)


def test_speed_strategies_share_the_declared_comparison_range():
    df = pd.DataFrame({
        "Physics_speed_kmh": [80, 90, 100, 105],
        "expert_optimal_speed": [100, 105, 110, 110],
    })
    branch = {"all_of": [
        _requirement(["speed_comparison_range"], "find_speed_expert_faster")["any_of"][0]["all_of"][0],
        _requirement(["speed_comparison_range"], "find_speed_peak_gap", "eq", 17.5)["any_of"][0]["all_of"][0],
        _requirement(["speed_comparison_range"], "find_speed_gap_closing")["any_of"][0]["all_of"][0],
    ]}

    result = _evaluate({"enabled": True, "any_of": [branch]}, df)

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(0, 3)


def test_trajectory_strategy_uses_declared_phase_range(monkeypatch):
    received = []
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_trajectory_offset",
        lambda segment: received.append(segment.index.tolist()) or np.array([2.0, 1.0, 0.5]),
    )
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(8)))

    result = deterministic.evaluate_requirements(
        _requirement(["trajectory_comparison_range"], "find_trajectory_convergence"),
        context,
        InclusiveRange(3, 5),
    )

    assert result.matched
    assert received == [[3, 4, 5]]


def test_trajectory_position_uses_one_meter_alignment_tolerance(monkeypatch):
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(3)))
    range_ = InclusiveRange(0, 2)

    for offsets, expected in (
        ([-1.0, -1.0, -1.0], "aligned"),
        ([1.0, 1.0, 1.0], "aligned"),
        ([-1.01, -1.01, -1.01], "tighter"),
        ([1.01, 1.01, 1.01], "wider"),
    ):
        monkeypatch.setattr(
            deterministic_facts,
            "_trajectory",
            lambda _context, _range, offsets=offsets: np.array(offsets),
        )

        assert deterministic_facts._trajectory_position(context, range_) == expected


def test_msp2_msp4_and_msp6_use_parent_segment_trajectory_requirements():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    assert requirements["MSP2"] == _requirement(
        {"start": "segment_start_iloc", "end": "segment_apex_iloc"},
        "find_trajectory_split",
        value="wider",
    )
    assert requirements["MSP4"] == _requirement(
        {"start": "segment_apex_iloc", "end": "segment_end_iloc"},
        "find_trajectory_split",
        value="wider",
    )
    assert requirements["MSP6"] == _requirement(
        {"start": "segment_start_iloc", "end": "segment_apex_iloc"},
        "find_trajectory_split",
        value="narrower",
    )


def test_trajectory_start_rules_preserve_range_evidence_and_classify_at_landmark():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    for label, range_tag, iloc_tag, position in (
        ("MSP9", "corner_entry_range", "segment_start_iloc", "tighter"),
        ("MSP33", "corner_entry_range", "segment_start_iloc", "wider"),
        ("MSP11", "corner_exit_range", "segment_apex_iloc", "tighter"),
        ("MSP16", "corner_exit_range", "segment_apex_iloc", "wider"),
    ):
        evidence = {
            "inputs": {"tags": [range_tag]},
            "condition": {},
        }
        classification = _requirement(
            [iloc_tag],
            "find_trajectory_position_at_iloc",
            value=position,
        )["any_of"][0]["all_of"][0]

        assert requirements[label] == {
            "enabled": True,
            "any_of": [{"all_of": [evidence, classification]}],
        }


def test_segment_landmark_inputs_resolve_to_parent_bounds_and_first_geometric_apex(
    monkeypatch,
):
    monkeypatch.setattr(
        deterministic_facts,
        "_shape_analysis",
        lambda *_args: {
            "phases": [
                {"entry": 3, "apex": 6, "exit": 8},
                {"entry": 9, "apex": 10, "exit": 11},
            ],
        },
    )
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(12)))
    scope = InclusiveRange(2, 11)

    segment_start = context.resolve_input(
        "segment_start_iloc", scope, deterministic.INPUT_REGISTRY,
    )
    segment_end = context.resolve_input(
        "segment_end_iloc", scope, deterministic.INPUT_REGISTRY,
    )
    segment_apex = context.resolve_input(
        "segment_apex_iloc", scope, deterministic.INPUT_REGISTRY,
    )

    assert segment_start == ResolvedInput(
        "segment_start_iloc", "iloc", 2, InclusiveRange(2, 2),
    )
    assert segment_end == ResolvedInput(
        "segment_end_iloc", "iloc", 11, InclusiveRange(11, 11),
    )
    assert segment_apex == ResolvedInput(
        "segment_apex_iloc", "iloc", 6, InclusiveRange(6, 6),
    )


def test_trajectory_split_classifies_aligned_start_to_wider_or_narrower_end(
    monkeypatch,
):
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(6)))
    inputs = [
        ResolvedInput("start", "iloc", 1, InclusiveRange(1, 1)),
        ResolvedInput("end", "iloc", 4, InclusiveRange(4, 4)),
    ]

    for offsets, expected in (
        ([np.nan, 1.0, np.nan, np.nan, 1.01, np.nan], "wider"),
        ([np.nan, -1.0, np.nan, np.nan, -1.01, np.nan], "narrower"),
    ):
        monkeypatch.setattr(
            deterministic_facts,
            "_corresponding_trajectory",
            lambda _context, offsets=offsets: np.array(offsets),
        )

        assert deterministic_facts._trajectory_split(context, inputs) == expected


def test_trajectory_split_fails_closed_when_either_endpoint_does_not_split(
    monkeypatch,
):
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(6)))
    inputs = [
        ResolvedInput("start", "iloc", 1, InclusiveRange(1, 1)),
        ResolvedInput("end", "iloc", 4, InclusiveRange(4, 4)),
    ]

    for offsets in (
        [0.0, 1.01, 0.0, 0.0, 2.0, 0.0],
        [0.0, -1.01, 0.0, 0.0, -2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, np.nan, 0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
    ):
        monkeypatch.setattr(
            deterministic_facts,
            "_corresponding_trajectory",
            lambda _context, offsets=offsets: np.array(offsets),
        )

        assert deterministic_facts._trajectory_split(context, inputs) is MISSING


def test_trajectory_split_requires_ordered_unique_resolvable_ilocs(monkeypatch):
    monkeypatch.setattr(
        deterministic_facts,
        "_corresponding_trajectory",
        lambda context: np.linspace(0.0, 2.0, len(context.telemetry)),
    )

    for index, start, end in (
        (range(6), 4, 1),
        (range(6), 1, 1),
        ([0, 1, 3, 4, 5], 1, 2),
        ([0, 1, 1, 2, 3], 1, 3),
    ):
        context = EvaluationContext.from_dataframe(pd.DataFrame(index=index))
        inputs = [
            ResolvedInput("start", "iloc", start, InclusiveRange(start, start)),
            ResolvedInput("end", "iloc", end, InclusiveRange(end, end)),
        ]

        assert deterministic_facts._trajectory_split(context, inputs) is MISSING

    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(6)))
    valid_inputs = [
        ResolvedInput("start", "iloc", 1, InclusiveRange(1, 1)),
        ResolvedInput("end", "iloc", 4, InclusiveRange(4, 4)),
    ]
    for offsets in (None, np.array([0.0, 1.0])):
        monkeypatch.setattr(
            deterministic_facts,
            "_corresponding_trajectory",
            lambda _context, offsets=offsets: offsets,
        )
        assert (
            deterministic_facts._trajectory_split(context, valid_inputs)
            is MISSING
        )


def test_msp2_matches_aligned_start_to_wider_apex_without_intermediate_samples(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 7}]},
    )
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: np.array([
            0.0, 1.0, np.nan, np.nan, np.nan, 1.4, 1.5, 1.5,
        ]),
    )
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(8)))

    result = deterministic.evaluate_labels(
        ["MSP2", "MSP33"], context, InclusiveRange(1, 7),
    )

    assert result.labels == ["MSP2"]
    assert (
        result.evaluations["MSP2"].matched_branches[0].evidence_range
        == InclusiveRange(1, 5)
    )
    assert not result.evaluations["MSP33"].matched


def test_trajectory_start_rules_ignore_later_divergence_or_recovery(monkeypatch):
    phases = [{"entry": 2, "apex": 5, "exit": 8}]
    trajectory = {"offsets": None}
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": phases},
    )
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: trajectory["offsets"],
    )

    for label, point, point_offset, later_offset, expected_range in (
        ("MSP9", 1, -1.01, 2.0, InclusiveRange(1, 5)),
        ("MSP33", 1, 1.01, -2.0, InclusiveRange(1, 5)),
        ("MSP11", 5, -1.01, 2.0, InclusiveRange(5, 8)),
        ("MSP16", 5, 1.01, -2.0, InclusiveRange(5, 8)),
    ):
        requirements = deterministic._requirements_for(
            label, deterministic.get_label(label),
        )
        for subsequent_offset in (later_offset, 0.0):
            offsets = np.full(10, subsequent_offset)
            offsets[point] = point_offset
            trajectory["offsets"] = offsets

            result = _evaluate(
                requirements, pd.DataFrame(index=range(10)), start=1, end=9,
            )

            assert result.matched, (label, subsequent_offset)
            assert (
                result.matched_branches[0].evidence_range == expected_range
            )


def test_trajectory_start_rules_fail_closed_for_invalid_landmark_offsets(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {
            "phases": [{"entry": 2, "apex": 5, "exit": 8}],
        },
    )
    trajectory = {"offsets": None}
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: trajectory["offsets"],
    )

    for label, point, matching_offset in (
        ("MSP9", 1, -1.01),
        ("MSP33", 1, 1.01),
        ("MSP11", 5, -1.01),
        ("MSP16", 5, 1.01),
    ):
        requirements = deterministic._requirements_for(
            label, deterministic.get_label(label),
        )
        for point_offset in (0.0, -matching_offset, np.nan):
            offsets = np.full(10, matching_offset)
            offsets[point] = point_offset
            trajectory["offsets"] = offsets

            assert not _evaluate(
                requirements, pd.DataFrame(index=range(10)), start=1, end=9,
            ).matched, (label, point_offset)

        for offsets in (None, np.full(9, matching_offset)):
            trajectory["offsets"] = offsets
            assert not _evaluate(
                requirements, pd.DataFrame(index=range(10)), start=1, end=9,
            ).matched, label

        unresolved_index = [iloc for iloc in range(10) if iloc != point]
        trajectory["offsets"] = np.full(len(unresolved_index), matching_offset)
        assert not _evaluate(
            requirements,
            pd.DataFrame(index=unresolved_index),
            start=1,
            end=9,
        ).matched, label


def test_trajectory_start_rules_require_range_evidence(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {
            "phases": [{"entry": None, "apex": 5, "exit": None}],
        },
    )
    trajectory = {"offsets": None}
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: trajectory["offsets"],
    )

    for label, range_tag, point, matching_offset in (
        ("MSP9", "corner_entry_range", 1, -1.01),
        ("MSP33", "corner_entry_range", 1, 1.01),
        ("MSP11", "corner_exit_range", 5, -1.01),
        ("MSP16", "corner_exit_range", 5, 1.01),
    ):
        offsets = np.zeros(10)
        offsets[point] = matching_offset
        trajectory["offsets"] = offsets
        requirements = deterministic._requirements_for(
            label, deterministic.get_label(label),
        )

        result = _evaluate(
            requirements, pd.DataFrame(index=range(10)), start=1, end=9,
        )

        assert not result.matched, label
        assert result.failed == [
            f"evidence: unavailable (missing input {range_tag})",
        ]


def test_msp2_fails_closed_for_non_wider_or_unavailable_splits(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 7}]},
    )
    requirements = deterministic._requirements_for(
        "MSP2", deterministic.get_label("MSP2"),
    )
    cases = (
        np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.4, 0.6, 1.1, -1.4, 0.0, 0.0]),
        np.array([0.0, 1.01, 0.4, 0.6, 1.1, 1.4, 0.0, 0.0]),
        np.array([0.0, -1.01, 0.4, 0.6, 1.1, 1.4, 0.0, 0.0]),
        np.array([0.0, np.nan, 0.4, 0.6, 1.1, 1.4, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.4, 0.6, 1.1, np.nan, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.4]),
        None,
    )

    for offsets in cases:
        monkeypatch.setattr(
            "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
            lambda _df, offsets=offsets: offsets,
        )
        assert not _evaluate(
            requirements, pd.DataFrame(index=range(8)), start=1, end=7,
        ).matched


def test_msp2_fails_closed_without_a_valid_geometric_apex(monkeypatch):
    requirements = deterministic._requirements_for(
        "MSP2", deterministic.get_label("MSP2"),
    )
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: np.linspace(0.0, 2.0, 8),
    )

    for phases in (
        [],
        [{"entry": 2, "apex": None, "exit": 7}],
        [{"entry": 2, "apex": 9, "exit": 10}],
        [{"entry": 1, "apex": 1, "exit": 7}],
    ):
        monkeypatch.setattr(
            "app.shared.annotation_agent_tools.measure_segment_shape",
            lambda *_args, phases=phases: {"phases": phases},
        )
        assert not _evaluate(
            requirements, pd.DataFrame(index=range(8)), start=1, end=7,
        ).matched


def test_msp4_matches_only_for_aligned_apex_to_wider_parent_end(monkeypatch):
    phases = [{"entry": 2, "apex": 5, "exit": 7}]
    offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.01])
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": phases},
    )
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_corresponding_trajectory_offset",
        lambda _df: offsets,
    )

    def evaluate():
        context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(8)))
        return deterministic.evaluate_labels(
            ["MSP4"], context, InclusiveRange(1, 7),
        )

    matched = evaluate()

    assert matched.labels == ["MSP4"]
    assert (
        matched.evaluations["MSP4"].matched_branches[0].evidence_range
        == InclusiveRange(5, 7)
    )

    for apex_offset, end_offset in (
        (1.01, 1.4),
        (0.0, 1.0),
        (0.0, -1.01),
        (np.nan, 1.4),
        (0.0, np.nan),
    ):
        offsets[5] = apex_offset
        offsets[7] = end_offset
        assert not evaluate().evaluations["MSP4"].matched

    offsets[5] = 0.0
    offsets[7] = 1.4
    phases.clear()
    assert not evaluate().evaluations["MSP4"].matched


def test_phase_resolver_uses_shape_landmarks(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.measure_segment_shape",
        lambda *_args: {"phases": [{"entry": 2, "apex": 5, "exit": 9}]},
    )
    result = _evaluate(
        _requirement(["corner_apex_range"], "find_phase_presence"),
        pd.DataFrame(index=range(12)),
        end=11,
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(3, 7)


def test_corresponding_trajectory_offset_uses_same_iloc_for_both_corner_directions():
    from app.shared.annotation_agent_tools import (
        calculate_corresponding_trajectory_offset,
    )

    for direction in (1.0, -1.0):
        angles = np.linspace(0.0, direction * np.pi / 2.0, 9)
        expert_x = 10.0 * np.cos(angles)
        expert_y = 10.0 * np.sin(angles)
        for radius_scale, expected_sign in ((1.2, 1.0), (0.8, -1.0)):
            player_x = expert_x.copy()
            player_y = expert_y.copy()
            player_x[-1] *= radius_scale
            player_y[-1] *= radius_scale
            offsets = calculate_corresponding_trajectory_offset(pd.DataFrame({
                "Graphics_player_pos_x": player_x,
                "Graphics_player_pos_y": player_y,
                "expert_optimal_player_pos_x": expert_x,
                "expert_optimal_player_pos_y": expert_y,
            }))

            assert offsets is not None
            assert float(offsets[-1]) * expected_sign > 1.0


def test_altitude_strategy_classifies_only_resolved_phase():
    df = pd.DataFrame({
        "expert_optimal_player_pos_x": [0.0, 1.0, 2.0, 3.0],
        "expert_optimal_player_pos_y": [0.0, 0.0, 0.0, 0.0],
        "expert_optimal_player_pos_z": [0.0, 0.1, 0.2, 1.0],
    })

    result = _evaluate(
        _requirement(["segment_range"], "find_entry_altitude_trend", value="uphill"),
        df,
    )

    assert result.matched


def test_altitude_strategy_uses_three_degree_threshold():
    x = np.arange(4, dtype=float)
    for angle, expected in (
        (4.0, "uphill"),
        (2.0, "level"),
        (-2.0, "level"),
        (-4.0, "downhill"),
    ):
        df = pd.DataFrame({
            "expert_optimal_player_pos_x": x,
            "expert_optimal_player_pos_y": np.zeros(4),
            "expert_optimal_player_pos_z": x * np.tan(np.radians(angle)),
        })

        result = _evaluate(
            _requirement(
                ["segment_range"], "find_entry_altitude_trend", value=expected,
            ),
            df,
        )

        assert result.matched, angle


def test_balance_and_grip_are_calculated_from_raw_tire_telemetry():
    size = 5
    df = pd.DataFrame({
        "Physics_slip_angle_front_left": [0.01] * size,
        "Physics_slip_angle_front_right": [0.01] * size,
        "Physics_slip_angle_rear_left": [0.2] * size,
        "Physics_slip_angle_rear_right": [0.2] * size,
        "Physics_slip_ratio_front_left": [2.0] * size,
        "Physics_slip_ratio_front_right": [2.0] * size,
        "Physics_slip_ratio_rear_left": [2.0] * size,
        "Physics_slip_ratio_rear_right": [2.0] * size,
    })
    branch = {"all_of": [
        _requirement(["control_range"], "find_oversteer")["any_of"][0]["all_of"][0],
        _requirement(["control_range"], "find_grip_over_limit")["any_of"][0]["all_of"][0],
    ]}

    assert _evaluate({"enabled": True, "any_of": [branch]}, df).matched


def _handling_balance_frame(balances, *, index=None):
    values = np.asarray(balances, dtype=float)
    front = np.maximum(-values, 0.0)
    rear = np.maximum(values, 0.0)
    return pd.DataFrame({
        "Physics_slip_angle_front_left": front,
        "Physics_slip_angle_front_right": front,
        "Physics_slip_angle_rear_left": rear,
        "Physics_slip_angle_rear_right": rear,
    }, index=index)


def _evaluate_handling_balance(df, onset, end):
    requirements = _requirement(
        ["onset", "end"],
        "find_oversteer_or_understeer_between_ilocs",
    )
    values = {"onset": onset, "end": end}
    inputs = InputRegistry({
        tag: InputDefinition(
            "iloc",
            lambda _context, _scope, tag=tag, value=value: ResolvedInput(
                tag, "iloc", value, InclusiveRange(value, value),
            ),
        )
        for tag, value in values.items()
    })
    scope = InclusiveRange(int(df.index.min()), int(df.index.max()))
    return RequirementInterpreter(inputs, deterministic.FACT_REGISTRY).evaluate(
        requirements,
        EvaluationContext(df),
        scope,
    )


def test_oversteer_or_understeer_between_ilocs_detects_new_threshold_entries():
    cases = (
        ([0.0, 0.03], 0, 1),
        ([0.0, -0.03], 0, 1),
        ([0.02, 0.03], 0, 1),
        ([-0.02, -0.03], 0, 1),
        ([0.0, 0.03, 0.0], 0, 2),
        ([0.0, -0.03, 0.0], 0, 2),
        ([0.03, 0.0, 0.03], 1, 2),
        ([-0.03, 0.0, -0.03], 1, 2),
    )

    for balances, onset, end in cases:
        result = _evaluate_handling_balance(
            _handling_balance_frame(balances), onset, end,
        )

        assert result.matched, (balances, onset, end)
        assert result.matched_branches[0].evidence_range == InclusiveRange(
            min(onset, end), max(onset, end),
        )


def test_oversteer_or_understeer_between_ilocs_uses_positional_predecessor():
    df = _handling_balance_frame([0.0, 0.03], index=[10, 20])

    result = _evaluate_handling_balance(df, 20, 20)

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(20, 20)


def test_oversteer_or_understeer_between_ilocs_accepts_reversed_endpoints():
    df = _handling_balance_frame([0.0, 0.03, 0.0], index=[10, 20, 30])

    result = _evaluate_handling_balance(df, 30, 20)

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(20, 30)


def test_oversteer_or_understeer_between_ilocs_rejects_carried_in_problems():
    cases = (
        [0.03, 0.03, 0.03, 0.03],
        [0.05, 0.04, 0.03, 0.025],
        [0.03, 0.04, 0.05, 0.06],
        [-0.03, -0.03, -0.03, -0.03],
        [-0.05, -0.04, -0.03, -0.025],
        [-0.03, -0.04, -0.05, -0.06],
    )

    for balances in cases:
        result = _evaluate_handling_balance(
            _handling_balance_frame(balances), 1, 3,
        )

        assert not result.matched, balances


def test_oversteer_or_understeer_between_ilocs_honors_boundaries_and_adjacency():
    rejected = (
        [0.0, 0.02],
        [0.0, -0.02],
        [0.0, np.nan, 0.03],
        [0.0, np.nan, -0.03],
        [0.03, 0.0],
        [-0.03, 0.0],
    )
    matched = (
        [0.0, np.nan, 0.0, 0.03],
        [0.0, np.nan, 0.0, -0.03],
    )

    for balances in rejected:
        result = _evaluate_handling_balance(
            _handling_balance_frame(balances), 0, len(balances) - 1,
        )
        assert not result.matched, balances
    for balances in matched:
        result = _evaluate_handling_balance(
            _handling_balance_frame(balances), 0, len(balances) - 1,
        )
        assert result.matched, balances


def test_oversteer_or_understeer_between_ilocs_rejects_missing_or_insufficient_data():
    complete = _handling_balance_frame([0.0, 0.0])
    missing = complete.drop(columns=["Physics_slip_angle_rear_right"])
    empty_range = _handling_balance_frame([0.0, 0.03], index=[0, 10])
    no_predecessor = _handling_balance_frame([0.03], index=[10])
    no_comparable_pair = _handling_balance_frame([np.nan, 0.0])

    assert not _evaluate_handling_balance(complete, 0, 1).matched
    assert not _evaluate_handling_balance(missing, 0, 1).matched
    assert not _evaluate_handling_balance(empty_range, 3, 5).matched
    assert not _evaluate_handling_balance(no_predecessor, 10, 10).matched
    assert not _evaluate_handling_balance(no_comparable_pair, 0, 1).matched


def test_opponent_strategies_share_one_cached_analysis(monkeypatch):
    calls = {"count": 0}

    def classify(*_args):
        calls["count"] += 1
        return {
            "outcome": "pass_completed",
            "confidence_level": "high",
            "candidates": [{
                "entry_signed_long_gap_m": 8.0,
                "exit_signed_long_gap_m": 2.0,
                "side_by_side_iloc_count": 2,
            }],
        }

    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.classify_opponent_interaction", classify,
    )
    branch = {"all_of": [
        _requirement(["opponent_interaction_range"], "find_opponent_outcome", value="pass_completed")["any_of"][0]["all_of"][0],
        _requirement(["opponent_interaction_range"], "find_opponent_gap_shrank")["any_of"][0]["all_of"][0],
    ]}

    result = _evaluate(
        {"enabled": True, "any_of": [branch]}, pd.DataFrame(index=range(5)),
    )

    assert result.matched
    assert calls["count"] == 1


def test_lap_catalog_is_interpreted_against_registered_time_strategies(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.run_pipeline_query",
        lambda *_args: ({
            "extra": {
                "delta_value": -40.0,
                "thresholds": {"label_significant_at_abs_delta": 50.0},
            },
        }, None),
    )
    requirements = deterministic._requirements_for(
        "EA", deterministic.get_label("EA"),
    )
    result = _evaluate(
        requirements,
        pd.DataFrame({"expert_time_difference": [50, 40, 30, 20, 10, 0]}),
    )

    assert result.matched
    assert result.branch == 1


def test_actual_sub_label_catalog_drives_detailed_range(monkeypatch):
    monkeypatch.setattr(
        "app.shared.annotation_agent_tools.calculate_trajectory_offset",
        lambda segment: np.linspace(2.0, 0.1, len(segment)),
    )

    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(8)),
        parent_start=0,
        parent_end=7,
        parent_main_labels=["RM"],
    )

    assert "RM7" in result.final_labels
    rm7 = next(
        item for item in result.label_annotations if item["label_id"] == "RM7"
    )
    assert (rm7["start_index"], rm7["end_index"]) == (0, 7)


def test_disabled_label_never_matches():
    result = _evaluate(
        {"enabled": False, "any_of": []}, pd.DataFrame(index=range(2)),
    )

    assert not result.matched
    assert result.failed == ["label disabled"]


def test_exclusive_matches_are_suppressed(monkeypatch):
    requirements = _requirement(["section_range"], "find_phase_presence")
    monkeypatch.setattr(deterministic, "get_label", lambda label_id: {
        "id": label_id,
        "selection_requirements": requirements,
        "exclusive_with": ["B"] if label_id == "A" else ["A"],
    })
    context = EvaluationContext.from_dataframe(pd.DataFrame(index=range(2)))

    result = deterministic.evaluate_labels(
        ["A", "B"], context, InclusiveRange(0, 1),
    )

    assert result.labels == []
    assert result.conflicts == [("A", "B")]


def test_same_label_overlaps_merge_transitively_but_other_labels_stay_separate():
    merged = deterministic._merge_label_annotations([
        {"label_id": "RM7", "start_index": 1, "end_index": 3, "passed": ["a"]},
        {"label_id": "RM7", "start_index": 3, "end_index": 5, "passed": ["b"]},
        {"label_id": "RM7", "start_index": 4, "end_index": 8, "passed": ["c"]},
        {"label_id": "MSP1", "start_index": 2, "end_index": 6, "passed": ["d"]},
    ])

    assert [
        (item["label_id"], item["start_index"], item["end_index"])
        for item in merged
    ] == [("RM7", 1, 8), ("MSP1", 2, 6)]


def test_detailed_segment_types_are_evaluated_after_discovery_on_parent_scope(
    monkeypatch,
):
    scopes = []

    def fake_evaluate(label_ids, _context, scope):
        scopes.append((tuple(label_ids), scope))
        if "RM7" in label_ids:
            branch = _branch(0, 2, 6, "trajectory: True")
            return deterministic.LabelEvaluation(
                ["RM7"], {"RM7": _matched(branch)},
            )
        branch = _branch(0, scope.start, scope.end, "shape: straight")
        return deterministic.LabelEvaluation(
            ["ST2"], {"ST2": _matched(branch)},
        )

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(10)),
        parent_start=0,
        parent_end=9,
        parent_main_labels=["RM"],
    )

    assert result.final_labels == ["RM7", "ST2"]
    sub_call = next(index for index, (labels, _scope) in enumerate(scopes) if "RM7" in labels)
    type_call = next(index for index, (labels, _scope) in enumerate(scopes) if "ST2" in labels)
    assert sub_call < type_call
    assert scopes[type_call][1] == InclusiveRange(0, 9)
    assert {
        (item["label_id"], item["start_index"], item["end_index"])
        for item in result.label_annotations
    } == {("RM7", 2, 6), ("ST2", 2, 6)}


def test_detailed_segment_type_range_must_contain_finalized_range(
    monkeypatch,
):
    sub_labels = {
        label_id: _matched(_branch(0, 8, 29, "sub-label evidence: True"))
        for label_id in ("RM1", "RM5", "RM7")
    }
    segment_types = {
        "ST1": _matched(_branch(0, 0, 39, "segment shape: in_corner")),
        "ST9": _matched(_branch(0, 8, 29, "corner shape: decreasing_radius")),
        "ST14": _matched(_branch(0, 16, 24, "entry altitude: downhill")),
        "ST17": _matched(_branch(0, 4, 20, "apex altitude: downhill")),
        "ST20": _matched(_branch(0, 29, 35, "exit altitude: downhill")),
    }

    def fake_evaluate(label_ids, _context, _scope):
        evaluations = sub_labels if "RM1" in label_ids else segment_types
        return deterministic.LabelEvaluation(list(evaluations), evaluations)

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(40)),
        parent_start=0,
        parent_end=39,
        parent_main_labels=["RM"],
    )

    ranges = {
        item["label_id"]: (item["start_index"], item["end_index"])
        for item in result.label_annotations
    }
    assert ranges == {
        "RM1": (8, 29),
        "RM5": (8, 29),
        "RM7": (8, 29),
        "ST1": (8, 29),
        "ST9": (8, 29),
    }
    assert not {"ST14", "ST17", "ST20"} & set(result.final_labels)


def test_detailed_segment_type_emits_once_on_finalized_range(monkeypatch):
    def fake_evaluate(label_ids, _context, _scope):
        if "RM7" in label_ids:
            branch = _branch(0, 0, 9, "trajectory: True")
            return deterministic.LabelEvaluation(
                ["RM7"], {"RM7": _matched(branch)},
            )
        branches = [_branch(0, 0, 9), _branch(1, 6, 8)]
        return deterministic.LabelEvaluation(
            ["ST2"], {"ST2": _matched(*branches)},
        )

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(10)),
        parent_start=0,
        parent_end=9,
        parent_main_labels=["RM"],
    )

    assert [
        (item["start_index"], item["end_index"])
        for item in result.label_annotations
        if item["label_id"] == "ST2"
    ] == [(0, 9)]


def test_detailed_segment_type_conflicts_do_not_remove_sub_label(monkeypatch):
    def fake_evaluate(label_ids, _context, _scope):
        if "RM7" in label_ids:
            branch = _branch(0, 2, 6, "trajectory: True")
            return deterministic.LabelEvaluation(
                ["RM7"], {"RM7": _matched(branch)},
            )
        segment_evaluations = {
            "ST1": _matched(_branch(0, 1, 7, "shape: in_corner")),
            "ST14": _matched(_branch(0, 0, 8, "entry altitude: downhill")),
            "ST7": _matched(_branch(0, 2, 6, "radius: constant")),
            "ST8": _matched(_branch(0, 2, 6, "radius: increasing")),
        }
        return deterministic.LabelEvaluation(
            ["ST1", "ST14"],
            segment_evaluations,
            [("ST7", "ST8")],
        )

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(9)),
        parent_start=0,
        parent_end=8,
        parent_main_labels=["RM"],
    )

    assert {
        (item["label_id"], item["start_index"], item["end_index"])
        for item in result.label_annotations
    } == {
        ("RM7", 2, 6),
        ("ST1", 2, 6),
        ("ST14", 2, 6),
    }
    assert "Suppressed 1 exclusive conflict(s)." in result.final_reasoning


def test_segment_type_alone_does_not_create_detailed_subsegment(monkeypatch):
    monkeypatch.setattr(
        deterministic,
        "evaluate_labels",
        lambda *_args: deterministic.LabelEvaluation([], {}),
    )

    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(6)),
        parent_start=0,
        parent_end=5,
        parent_main_labels=["RM"],
    )

    assert result.final_labels == []
    assert result.label_annotations == []


def test_saved_children_filter_exact_branch_range_before_merge(monkeypatch):
    def fake_evaluate(label_ids, _context, _scope):
        if "RM7" in label_ids:
            branches = [_branch(0, 1, 4), _branch(1, 3, 7)]
            return deterministic.LabelEvaluation(
                ["RM7"], {"RM7": _matched(*branches)},
            )
        return deterministic.LabelEvaluation([], {})

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)

    result = deterministic.calculate_detailed_annotation(
        pd.DataFrame(index=range(9)),
        parent_start=0,
        parent_end=8,
        parent_main_labels=["RM"],
        existing_children=[{
            "start_index": 1, "end_index": 4, "labels": ["RM7"],
        }],
    )

    assert [
        (item["start_index"], item["end_index"])
        for item in result.label_annotations
    ] == [(3, 7)]


def test_lap_contract_uses_branch_evidence_for_reasoning(monkeypatch):
    monkeypatch.setattr(
        deterministic,
        "_resolve_circuit_sections",
        lambda *_args: ("silverstone1", ["silverstone1"]),
    )
    monkeypatch.setattr(deterministic, "_is_far_from_expert_in_pit", lambda *_args: False)

    evaluated_label_groups = []

    def fake_evaluate(label_ids, _context, _scope):
        evaluated_label_groups.append(tuple(label_ids))
        if "EA" in label_ids:
            branch = _branch(0, 2, 5, "time: falling")
            return deterministic.LabelEvaluation(["EA"], {"EA": _matched(branch)})
        return deterministic.LabelEvaluation([], {})

    monkeypatch.setattr(deterministic, "evaluate_labels", fake_evaluate)
    result = deterministic.calculate_lap_annotation(
        pd.DataFrame(index=range(8)),
        lap_start=0,
        lap_end=7,
        section_id="silverstone1",
        section_start=0,
        section_end=7,
        circuit_id="silverstone",
    )

    assert result.label_ids == ["silverstone", "silverstone1", "EA"]
    assert not any(
        label.startswith("ST")
        for label_ids in evaluated_label_groups
        for label in label_ids
    )
    assert "iloc range [2, 5]" in result.reasoning


def test_public_pipeline_returns_deterministic_lap_contract(monkeypatch):
    expected = deterministic.LapAnnotationResult(
        section_id="silverstone1",
        start_index=0,
        end_index=4,
        label_ids=["EA"],
        reasoning="ok",
        submitted=True,
        rejected_proposals=[],
        transcript="deterministic label evaluation",
        tool_calls=0,
    )
    monkeypatch.setattr(
        workflow, "calculate_lap_annotation", lambda *_args, **_kwargs: expected,
    )

    result = workflow.run_annotation(
        flow="lap",
        df=pd.DataFrame(index=range(5)),
        config=workflow.AnnotationPipelineConfig(provider_id="deterministic"),
        lap_start=0,
        lap_end=4,
        section_id="silverstone1",
        section_start=0,
        section_end=4,
        circuit_id="silverstone",
    )

    assert result is expected
