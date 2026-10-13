import pandas as pd
import pytest

from training.local_annotation_agent.workflow import deterministic
from training.local_annotation_agent.workflow.deterministic_engine import (
    HalfOpenRange,
    validate_requirements,
)
from training.local_annotation_agent.workflow.deterministic_facts import EvaluationContext
from training.shared import annotation_agent_tools


SLIP_COLUMNS = [
    f"Physics_{signal}_{wheel}"
    for signal in ("slip_angle", "slip_ratio")
    for wheel in ("front_left", "front_right", "rear_left", "rear_right")
] + ["driver_push_to_limit", "slip_balance"]


@pytest.fixture
def telemetry():
    return pd.DataFrame({
        "Physics_speed_kmh": [100.0] * 5,
        "expert_optimal_speed": [110.0] * 5,
        "expert_time_difference": [0.0, 200.0, 400.0, 600.0, 800.0],
        **{column: [0.0, 0.0, 2.0, 2.0, 2.0] for column in SLIP_COLUMNS},
    }, index=range(10, 15))


def test_annotation_context_excludes_slip_without_changing_source(telemetry):
    original = telemetry.copy(deep=True)

    context = EvaluationContext.from_dataframe(telemetry)

    assert not set(SLIP_COLUMNS).intersection(context.telemetry.columns)
    assert context.telemetry.index.equals(telemetry.index)
    assert context.telemetry["Physics_speed_kmh"].tolist() == [100.0] * 5
    pd.testing.assert_frame_equal(telemetry, original)


def test_slip_dependent_labels_are_disabled(telemetry):
    labels = ["MSP15", "MSP20", "MSP42", "MSP43", "MSP44", "MSP45", "MSP46", "MSP47"]
    result = deterministic.evaluate_labels(
        labels, EvaluationContext(telemetry), HalfOpenRange(10, 15),
    )

    assert result.labels == []
    for label in labels:
        assert result.evaluations[label].failed == ["label disabled"]
        requirements = deterministic._requirements_for(label, None)
        assert requirements == {
            "enabled": False, "any_of": [],
        }
        assert validate_requirements(
            requirements, deterministic.INPUT_REGISTRY, deterministic.FACT_REGISTRY,
        ) == []
    assert not {
        "find_oversteer_or_understeer_between_ilocs",
        "find_oversteer", "find_understeer",
        "find_grip_over_limit", "find_sustained_low_grip",
    }.intersection(deterministic.FACT_REGISTRY.names())


@pytest.mark.parametrize("column", SLIP_COLUMNS)
def test_annotation_queries_cannot_read_slip_columns(telemetry, column):
    result, error = annotation_agent_tools.run_pipeline_query(
        telemetry, "find_extremum",
        {"range": [10, 15], "column": column, "kind": "max"},
    )

    assert error
    assert result["value"] is None


def test_annotation_queries_still_read_supported_columns(telemetry):
    result, error = annotation_agent_tools.run_pipeline_query(
        telemetry, "find_extremum",
        {"range": [10, 15], "column": "Physics_speed_kmh", "kind": "max"},
    )

    assert error is None
    assert result["value"] == 100.0
    assert result["iloc"] == 10


def test_slip_graphs_are_unavailable_even_with_precomputed_data(telemetry):
    removed = ["trajectory_balance", "push_limit"]
    graph_ids = {graph["id"] for graph in annotation_agent_tools.AGENT_GRAPH_DEFINITIONS}

    assert not graph_ids.intersection(removed)
    for graph_id in removed:
        assert annotation_agent_tools.build_graph(graph_id, telemetry) is None
    assert annotation_agent_tools.generate_telemetry_graphs(
        telemetry, 10, 15, graph_ids=removed,
    ) == []
    assert annotation_agent_tools.render_graph_builds({
        "trajectory_balance": telemetry[["slip_balance"]],
        "push_limit": telemetry[["driver_push_to_limit"]],
    }, 10, 15) == []
    assert annotation_agent_tools.build_graph("speed", telemetry) is not None


@pytest.mark.parametrize("flow", ["lap", "detailed"])
def test_annotation_results_are_independent_of_slip_values(telemetry, flow):
    if flow == "lap":
        calculate = deterministic.calculate_lap_annotation
        kwargs = dict(
            lap_start=10, lap_end=15, section_id="", section_start=10,
            section_end=15, circuit_id="",
        )
    else:
        calculate = deterministic.calculate_detailed_annotation
        kwargs = dict(parent_start=10, parent_end=15, parent_main_labels=["MSP"])

    assert calculate(telemetry, **kwargs) == calculate(
        telemetry.drop(columns=SLIP_COLUMNS), **kwargs,
    )
