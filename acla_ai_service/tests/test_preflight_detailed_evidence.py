from app.local_annotation_agent.workflow.preflight_detailed import _event_sentence


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
