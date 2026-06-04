from app.voice.pipecat_pipeline import _format_observation_for_llm


def test_attack_window_observation_formats_radio_prompt() -> None:
    text = _format_observation_for_llm({
        "event": "attack_window",
        "projected_section": "Clearways",
        "time_to_overlap_seconds": 3.2,
        "closing_speed_mps": 4.4,
        "distance_m": 18.0,
        "opponent_id": 12,
    })

    assert "attack_window at Clearways" in text
    assert "arriving in 3.2s" in text
    assert "Tell the driver an attack is opening" in text


def test_defense_threat_observation_formats_radio_prompt() -> None:
    text = _format_observation_for_llm({
        "event": "defense_threat",
        "projected_section": "Surtees",
        "time_to_overlap_seconds": 2.5,
        "closing_speed_mps": 5.1,
        "distance_m": 14.0,
        "opponent_slot": 3,
    })

    assert "defense_threat at Surtees" in text
    assert "opponent 3" in text
    assert "Tell the driver to defend" in text
