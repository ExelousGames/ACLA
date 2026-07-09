import pandas as pd

from app.local_annotation_agent.workflow.preflight import (
    _preflight_point_similarity_summary,
)
from app.local_annotation_agent.workflow.preflight_lap import (
    LAP_PREFLIGHT_QUERY_SPECS,
)
from app.shared.annotation_agent_tools import build_graph, run_pipeline_query


def test_point_similarity_query_returns_simple_control_similarity():
    df = pd.DataFrame(
        {
            "expert_optimal_throttle": [0.0, 0.5, 0.5, 0.4],
            "Physics_gas": [0.0, 0.5, 1.0, 0.4],
        },
        index=[10, 11, 12, 13],
    )
    table = build_graph("throttle", df)

    payload, error = run_pipeline_query(
        table,
        "measure_point_similarity",
        {
            "range": [10, 13],
            "player_column": "Physics_gas",
            "expert_column": "expert_optimal_throttle",
            "smoothing_window": 1,
        },
    )

    assert error is None
    assert payload["iloc"] == 13
    assert payload["value"] == 0.88
    assert payload["extra"]["sample_count"] == 4
    assert payload["samples"] is None
    assert _preflight_point_similarity_summary(
        "query_telemetry.measure_point_similarity.throttle",
        payload,
        {
            "player_column": "Physics_gas",
            "expert_column": "expert_optimal_throttle",
        },
    ) == "Driver throttle similarity to expert: 88%."


def test_lap_preflight_runs_throttle_and_brake_point_similarity():
    tool_ids = {
        spec["tool_id"]
        for spec in LAP_PREFLIGHT_QUERY_SPECS
        if spec.get("query_id") == "measure_point_similarity"
    }

    assert tool_ids == {
        "query_telemetry.measure_point_similarity.throttle",
        "query_telemetry.measure_point_similarity.brake",
    }
