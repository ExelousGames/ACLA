from app.shared.label_hierarchy import build_track_area_segments


def test_track_area_segments_clip_child_analysis_to_parent_area():
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

    assert [segment["parent_segment_id"] for segment in segments] == [
        "brands_hatch2",
        "brands_hatch3",
    ]
    assert segments[0]["start_index"] == 0
    assert segments[0]["end_index"] == 4
    assert segments[0]["child_segments"][0]["start_index"] == 1
    assert segments[0]["child_segments"][0]["end_index"] == 4
    assert [label["label_id"] for label in segments[0]["child_segments"][0]["labels"]] == [
        "MSP",
        "MSP1",
    ]
    assert segments[1]["child_segments"][0]["start_index"] == 4
    assert segments[1]["child_segments"][0]["end_index"] == 7


def test_track_area_segments_fall_back_without_section_positions():
    segments = build_track_area_segments(
        [{"start_index": 0, "end_index": 3, "labels": ["MSP1"]}],
        [{"some_other_column": 1}],
        "brands_hatch",
    )

    assert len(segments) == 1
    assert segments[0]["main_label_id"] == "MSP"
    assert segments[0]["sub_segments"][0]["labels"][0]["label_id"] == "MSP1"
