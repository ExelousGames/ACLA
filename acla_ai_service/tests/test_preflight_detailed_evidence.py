import pandas as pd

from app.local_annotation_agent.workflow.preflight_detailed import (
    _corner_phase_ranges,
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


def test_speed_gap_evidence_omits_phase_ranges_without_speed_difference_shape():
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

    assert _corner_phase_ranges(121, 126, phases) == [
        ("entry", [121, 122]),
        ("apex", [123, 125]),
        ("exit", [126, 126]),
    ]
    assert "player faster than expert" not in event_names
    assert event_names == ["speed gap closing"]
    assert any(
        "the player speed gap moved from 13.818% slower than expert "
        "to 0.25% slower than expert" in sentence
        for sentence in sentences
    )
    assert not any("corner phase ranges were" in sentence for sentence in sentences)
    assert not any("internal" in sentence for sentence in sentences)


def test_speed_gap_shape_uses_speed_difference_slope_turns():
    df = pd.DataFrame(
        {
            "expert_optimal_speed": [100.0] * 18,
            "speed_difference": [
                10.0,
                10.0,
                2.0,
                -27.0,
                -35.0,
                -34.0,
                -21.0,
                -4.0,
                12.0,
                22.0,
                24.0,
                24.0,
                19.0,
                6.0,
                -2.0,
                -7.0,
                -11.0,
                -10.0,
            ],
        },
        index=list(range(33, 51)),
    )
    phases = [{"entry": 33, "apex": 40, "exit": 50}]

    events = _speed_events(df, 33, 50, {}, phases)
    sentences = [_event_sentence(event) for event in events]

    assert any(
        "speed_difference dip was across entry and apex" in sentence
        and "from iloc 35 to 40" in sentence
        for sentence in sentences
    )
    assert any(
        "speed_difference spike was across apex and exit" in sentence
        and "from iloc 40 to 46" in sentence
        for sentence in sentences
    )
    assert not any("internal" in sentence for sentence in sentences)
    assert not any("local" in sentence for sentence in sentences)
    assert not any("from iloc 48 to 50" in sentence for sentence in sentences)


def test_speed_gap_shape_requires_slope_turn():
    df = pd.DataFrame(
        {
            "expert_optimal_speed": [100.0] * 5,
            "speed_difference": [-10.0, -8.0, -6.0, -4.0, -2.0],
        },
        index=[10, 11, 12, 13, 14],
    )
    phases = [{"entry": 10, "apex": 12, "exit": 14}]

    events = _speed_events(df, 10, 14, {}, phases)
    sentences = [_event_sentence(event) for event in events]

    assert not any(
        "speed_difference spike" in sentence
        for sentence in sentences
    )
    assert not any(
        "speed_difference dip" in sentence
        for sentence in sentences
    )
