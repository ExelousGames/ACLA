from app.local_annotation_agent.workflow.preflight import (
    _preflight_lap_time_gap_slope_summary,
    _preflight_time_gap_slope_summary,
)


def _extra_with_slopes(slopes):
    runs = []
    value = 100.0
    for index, slope in enumerate(slopes):
        start_iloc = index * 10
        end_iloc = start_iloc + 10
        end_value = value + slope * 10
        runs.append({
            "direction": "rising" if end_value >= value else "falling",
            "start_iloc": start_iloc,
            "end_iloc": end_iloc,
            "start_value": value,
            "end_value": end_value,
            "slope": slope,
        })
        value = end_value
    return {
        "unit": "ms",
        "slope_unit": "ms/iloc",
        "point_trend_runs": runs,
    }


def test_lap_time_gap_end_summary_keeps_total_value_change():
    summary = _preflight_lap_time_gap_slope_summary(
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
    assert "starting slope" not in summary
    assert "Ending slope" not in summary


def test_shared_time_gap_summary_keeps_existing_endpoint_slope_wording():
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

    assert "starting slope" in summary
    assert "Ending slope" in summary


def test_lap_time_gap_slope_trend_requires_three_points():
    summary = _preflight_lap_time_gap_slope_summary(_extra_with_slopes([1.0, 2.0]))

    assert (
        "Time gap starting slope trend: unavailable because at least 3 local "
        "slope points are required and only 2 were available."
    ) in summary
    assert (
        "Time gap ending slope trend: unavailable because at least 3 local "
        "slope points are required and only 2 were available."
    ) in summary
    assert "Time gap ends at index 20" in summary


def test_lap_time_gap_starting_slope_trend_detects_raising():
    summary = _preflight_lap_time_gap_slope_summary(
        _extra_with_slopes([1.0, 2.0, 3.0, 3.0])
    )

    assert (
        "Time gap starting slope trend: raising based on the first 3 local slope points"
    ) in summary


def test_lap_time_gap_ending_slope_trend_detects_raising_then_flattening():
    summary = _preflight_lap_time_gap_slope_summary(
        _extra_with_slopes([4.0, 4.0, 1.0, 2.0, 2.1])
    )

    assert (
        "Time gap ending slope trend: raising then flattening "
        "based on the last 3 local slope points"
    ) in summary


def test_lap_time_gap_starting_slope_trend_detects_falling():
    summary = _preflight_lap_time_gap_slope_summary(
        _extra_with_slopes([3.0, 2.0, 1.0, 1.0])
    )

    assert (
        "Time gap starting slope trend: falling based on the first 3 local slope points"
    ) in summary


def test_lap_time_gap_ending_slope_trend_detects_falling_then_flattening():
    summary = _preflight_lap_time_gap_slope_summary(
        _extra_with_slopes([4.0, 4.0, 3.0, 1.0, 0.9])
    )

    assert (
        "Time gap ending slope trend: falling then flattening "
        "based on the last 3 local slope points"
    ) in summary
