from app.local_annotation_agent.workflow.preflight import (
    _preflight_time_gap_slope_summary,
)


def test_time_gap_end_summary_includes_total_change():
    summary = _preflight_time_gap_slope_summary(
        {
            "unit": "ms",
            "slope_unit": "ms/iloc",
            "start_slope": 1473,
            "end_slope": 463.99,
            "point_trend_runs": [
                {
                    "direction": "falling",
                    "start_iloc": 0,
                    "end_iloc": 6,
                    "start_value": -77854.34,
                    "end_value": -73854.1,
                    "slope": 1473,
                },
            ],
        }
    )

    assert (
        "Time gap ends at index 6 with value -73854.1 ms, "
        "higher than the starting value by 4000.24 ms total change (5.138%)."
    ) in summary
