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
    return {
        "enabled": True,
        "any_of": [{"all_of": [{
            "inputs": {"tags": list(tags)},
            "condition": {"fact": fact, "operator": operator, "value": value},
        }]}],
    }


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


def test_telemetry_is_smoothed_once_with_centered_three_sample_median():
    df = pd.DataFrame({"signal": [0.0, 0.0, 10.0, 0.0, 1.0, 1.0]})

    telemetry = smooth_telemetry(df)

    assert telemetry["signal"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
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
        pd.DataFrame({"expert_time_difference": [0.0, 100.0, 100.0]}),
    )

    assert result.matched
    assert len(calls) == 1


def test_catalog_requirements_are_valid():
    assert deterministic.validate_catalog() == []


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
    assert "player_brake_application_onset_iloc" in deterministic.INPUT_REGISTRY.names()
    assert "brake_comparison_range" in deterministic.INPUT_REGISTRY.names()
    assert "expert_upshift_range" in deterministic.INPUT_REGISTRY.names()
    assert "expert_downshift_range" in deterministic.INPUT_REGISTRY.names()
    assert "player_upshift_onset_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert "expert_upshift_end_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert "player_upshift_iloc" not in deterministic.INPUT_REGISTRY.names()
    assert not hasattr(deterministic, "KNOWN_FACTS")
    assert not hasattr(deterministic, "FactSet")


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


def test_missing_input_fails_closed_with_rejected_reason():
    result = _evaluate(
        _requirement(
            ["player_brake_release_end_iloc", "expert_brake_release_end_iloc"],
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


def test_compare_ilocs_preserves_declaration_order_and_uses_point_envelope():
    df = pd.DataFrame({
        "Physics_brake": [0, 0, 0, 0, 1, 1, 0],
        "expert_optimal_brake": [0, 1, 1, 0, 0, 0, 0],
    })
    result = _evaluate(
        _requirement(
            ["player_brake_application_onset_iloc", "expert_brake_application_onset_iloc"],
            "compare_ilocs", value="later",
        ),
        df,
    )

    assert result.matched
    assert result.matched_branches[0].evidence_range == InclusiveRange(1, 4)


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
        ("expert_upshift_range", InclusiveRange(2, 3)),
        ("expert_downshift_range", InclusiveRange(8, 11)),
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

        assert result.matched, label_id
        assert result.matched_branches[0].evidence_range == InclusiveRange(2, 3)


def test_shift_timing_aligns_matching_changes_and_rejects_ambiguous_changes():
    expert = [2, 2, 2, 3, 3, 3]
    early_requirements = deterministic._requirements_for(
        "MSP35", deterministic.get_label("MSP35"),
    )
    late_requirements = deterministic._requirements_for(
        "MSP36", deterministic.get_label("MSP36"),
    )
    aligned_requirements = _requirement(
        ["expert_upshift_range"], "compare_upshift_timing", value="aligned",
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
        "Physics_gear": [4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1],
        "expert_optimal_gear": [5, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    }, index=index)
    context = EvaluationContext.from_dataframe(df)
    evaluated = deterministic.evaluate_labels(
        ["MSP35", "MSP36", "MSP37", "MSP38"],
        context, InclusiveRange(33, 50),
    )

    assert evaluated.labels == ["MSP38"]
    branch = evaluated.evaluations["MSP38"].matched_branches[0]
    assert branch.evidence_range == InclusiveRange(35, 38)

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
    assert (annotation["start_index"], annotation["end_index"]) == (35, 38)
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


def test_msp22_uses_localized_brake_range_and_requires_all_endpoints(monkeypatch):
    landmarks = {
        "player": {
            "application_onset": 4,
            "release_end": 9,
            "peak": 0.9,
        },
        "expert": {
            "application_onset": 2,
            "release_end": 7,
            "peak": 0.7,
        },
    }
    scopes = []

    def fake_landmarks(_context, scope, driver, control):
        assert control == "brake"
        scopes.append(scope)
        return landmarks[driver]

    monkeypatch.setattr(deterministic_facts, "_control_landmarks", fake_landmarks)
    requirements = deterministic._requirements_for(
        "MSP22", deterministic.get_label("MSP22"),
    )

    matched = _evaluate(requirements, pd.DataFrame(index=range(12)), end=11)

    assert matched.matched
    assert matched.matched_branches[0].evidence_range == InclusiveRange(2, 9)
    assert InclusiveRange(2, 9) in scopes

    landmarks["expert"]["release_end"] = None
    missing_endpoint = _evaluate(
        requirements, pd.DataFrame(index=range(12)), end=11,
    )

    assert not missing_endpoint.matched
    assert "missing input" in missing_endpoint.failed[0]


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
        ("player", "brake"): {"release_onset": 8, "release_end": 12},
        ("expert", "brake"): {"release_onset": 4, "release_end": 8},
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


def test_every_control_onset_comparison_has_matching_end_comparison():
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
                tuple(predicate["inputs"]["tags"]): predicate["condition"]
                for predicate in predicates
            }
            for control in ("brake", "throttle"):
                for phase in ("application", "release"):
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
                    assert end_tags in by_tags, f"{label_id} omits {phase}_end"
                    onset = by_tags[onset_tags]
                    if onset["operator"] == "exists" or onset["value"] in {
                        "earlier", "later",
                    }:
                        assert by_tags[end_tags] == onset


def test_every_shift_timing_requirement_uses_its_expert_range():
    root = Path(__file__).parents[1] / "app/internal_knowledge_base"
    requirements = json.loads(
        (root / "sub_label_annotation.json").read_text(encoding="utf-8")
    )["sub_label_selection_requirements"]

    for label_id, direction, relation in (
        ("MSP35", "up", "earlier"),
        ("MSP36", "up", "later"),
        ("MSP37", "down", "earlier"),
        ("MSP38", "down", "later"),
    ):
        assert requirements[label_id] == _requirement(
            [f"expert_{direction}shift_range"],
            f"compare_{direction}shift_timing",
            value=relation,
        )


def test_range_fact_inspects_only_its_declared_range():
    df = pd.DataFrame({
        "Physics_speed_kmh": [0, 0, 100, 100, 0, 0],
        "expert_optimal_speed": [100, 100, 90, 90, 100, 100],
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
        _requirement(["speed_comparison_range"], "find_speed_peak_gap", "eq", 20.0)["any_of"][0]["all_of"][0],
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
    assert result.branch == 0


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
