import pandas as pd

from app.local_annotation_agent.workflow.preflight_detailed import (
    _event_sentence,
    _speed_events,
)


def test_brake_peak_speed_context_names_speed():
    sentence = _event_sentence(
        {
            "event": "peak brake pressure lower than expert",
            "phase": "unknown",
            "range": [58, 58],
            "measurements": {
                "player_value": 0.52,
                "expert_value": 0.67,
                "player_iloc": 58,
                "speed_gap_percent_at_player_peak": 7.479,
                "speed_gap_relation_at_player_peak": "faster",
            },
            "confidence": "strong",
        }
    )

    assert "the player speed was 7.479% faster than expert" in sentence
    assert "at the player brake peak" in sentence


def test_speed_gap_evidence_uses_phase_ranges_not_legacy_extrema():
    df = pd.DataFrame(
        {
            "Physics_speed_kmh": [86.182, 90.47, 95.0, 102.277, 98.0, 99.75],
            "expert_optimal_speed": [100.0] * 6,
        },
        index=[121, 122, 123, 124, 125, 126],
    )
    by_tool = {
        "query_telemetry.find_extremum.speed_difference.min": {
            "result": {"iloc": 124, "value": -2.277},
        },
    }
    phases = [{"entry": 121, "apex": 124, "exit": 126}]

    events = _speed_events(df, 121, 126, by_tool, phases)
    event_names = [event["event"] for event in events]
    sentences = [_event_sentence(event) for event in events]

    assert "player faster than expert" not in event_names
    assert "speed gap closing at entry" in event_names
    assert "speed gap closing at apex" in event_names
    assert any(
        "the player speed gap moved from 13.818% slower than expert "
        "to 2.277% faster than expert" in sentence
        for sentence in sentences
    )
