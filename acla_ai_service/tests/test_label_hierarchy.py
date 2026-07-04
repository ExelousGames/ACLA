from app.shared.label_hierarchy import build_track_area_segments


def test_track_area_segments_keep_classifier_child_range_under_overlapping_parent_area():
    telemetry_data = [
        {"Graphics_normalized_car_position": 0.02},
        {"Graphics_normalized_car_position": 0.04},
        {"Graphics_normalized_car_position": 0.06},
        {"Graphics_normalized_car_position": 0.08},
        {"Graphics_normalized_car_position": 0.12},
        {"Graphics_normalized_car_position": 0.14},
        {"Graphics_normalized_car_position": 0.16},
        {"Graphics_normalized_car_position": 0.17},
    ]
    raw_segments = [
        {
            "start_index": 1,
            "end_index": 7,
            "labels": ["MSP1"],
        },
    ]

    segments = build_track_area_segments(raw_segments, telemetry_data, "brands_hatch")

    assert [segment["parent_labels"] for segment in segments] == [
        ["brands_hatch2"],
        ["brands_hatch3"],
    ]
    assert segments[0]["start_index"] == 0
    assert segments[0]["end_index"] == 4
    assert segments[0]["child_segments"][0]["start_index"] == 1
    assert segments[0]["child_segments"][0]["end_index"] == 7
    assert segments[0]["child_segments"][0]["labels"] == [
        "MSP",
        "MSP1",
    ]
    assert segments[1]["child_segments"][0]["start_index"] == 1
    assert segments[1]["child_segments"][0]["end_index"] == 7


def test_track_area_segments_can_include_empty_lap_sections():
    telemetry_data = [
        {"Graphics_normalized_car_position": 0.02},
        {"Graphics_normalized_car_position": 0.04},
        {"Graphics_normalized_car_position": 0.06},
        {"Graphics_normalized_car_position": 0.08},
        {"Graphics_normalized_car_position": 0.12},
        {"Graphics_normalized_car_position": 0.14},
    ]
    raw_segments = [
        {
            "start_index": 4,
            "end_index": 6,
            "labels": ["MSP1"],
        },
    ]

    segments = build_track_area_segments(
        raw_segments,
        telemetry_data,
        "brands_hatch",
        include_empty_sections=True,
    )

    assert [segment["parent_labels"] for segment in segments] == [
        ["brands_hatch2"],
        ["brands_hatch3"],
    ]
    assert segments[0]["child_segments"] == []
    assert segments[1]["child_segments"][0]["labels"] == ["MSP", "MSP1"]


def test_track_area_segments_fall_back_without_section_positions():
    segments = build_track_area_segments(
        [{"start_index": 0, "end_index": 3, "labels": ["MSP1"]}],
        [{"some_other_column": 1}],
        "brands_hatch",
    )

    assert len(segments) == 1
    assert segments[0]["parent_labels"] == ["MSP"]
    assert segments[0]["child_segments"][0]["labels"][0] == "MSP1"
