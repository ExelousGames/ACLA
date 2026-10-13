from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from app.integrations.backend.client import BackendService
from app.shared import circuit_sections
from app.services.runtime_segment_splitter import split_runtime_segments


@pytest.fixture
def backend_map(monkeypatch):
    maps = [{
        "id": "map-1", "game": "acc", "circuit_name": "Custom Circuit",
        "source_track_key": "custom_layout",
    }]
    details = {"centerline_segments": [
        {"id": "backend-a", "name": "Backend Hairpin", "tags": ["corner"],
         "start_position": 0.1, "end_position": 0.3},
        {"id": "backend-b", "name": "Finish Straight", "tags": ["long straight"],
         "start_position": 0.9, "end_position": 0.1},
    ]}
    calls = []

    async def call_backend(_self, endpoint, **kwargs):
        calls.append(endpoint)
        return deepcopy({"list": maps} if endpoint.startswith("circuit-map/list") else details)

    monkeypatch.setattr(BackendService, "call_backend_function", call_backend)
    monkeypatch.setattr(circuit_sections, "time", SimpleNamespace(monotonic=lambda: 120))
    circuit_sections._cached_sections.cache_clear()
    yield maps, details, calls
    circuit_sections._cached_sections.cache_clear()


def telemetry(positions, track="custom_layout", game="acc"):
    return pd.DataFrame({
        "Graphics_normalized_car_position": positions,
        "Static_track": track,
        "game": game,
    })


def test_runtime_uses_backend_section_ids_names_and_ranges(backend_map):
    result = split_runtime_segments(telemetry([0.15, 0.25, 0.95, 0.05]), None)

    assert [(s["circuit_section_id"], s["circuit_section_name"],
             s["start_index"], s["end_index"]) for s in result["segments"]] == [
        ("backend-a", "Backend Hairpin", 0, 2),
        ("backend-b", "Finish Straight", 2, 3),
        ("backend-b", "Finish Straight", 3, 4),
    ]
    assert result["segments"][0]["normalized_position_range"] == [0.1, 0.3]


@pytest.mark.parametrize("track", ["CUSTOM LAYOUT", "Custom Circuit"])
def test_map_matches_source_key_or_display_name(backend_map, track):
    assert circuit_sections.get_circuit_sections(telemetry([0.15], track))[0]["id"] == "backend-a"


def test_source_key_takes_precedence_over_display_name(backend_map):
    maps, _, calls = backend_map
    maps.insert(0, {"id": "wrong-map", "circuit_name": "custom_layout", "source_track_key": "other"})
    circuit_sections.get_circuit_sections(telemetry([0.15]))
    assert calls[-1] == "circuit-map/map-1"


def test_map_edits_refresh_names_and_ranges(monkeypatch, backend_map):
    _, details, calls = backend_map
    frame = telemetry([0.15])
    assert circuit_sections.get_circuit_sections(frame)[0]["name"] == "Backend Hairpin"
    circuit_sections.get_circuit_sections(frame)
    assert len(calls) == 2
    details["centerline_segments"][0].update(name="Renamed Corner", end_position=0.4)
    monkeypatch.setattr(circuit_sections, "time", SimpleNamespace(monotonic=lambda: 180))
    section = circuit_sections.get_circuit_sections(frame)[0]
    assert section["name"] == "Renamed Corner"
    assert section["normalized_position_range"] == (0.1, 0.4)


@pytest.mark.asyncio
async def test_lookup_works_inside_async_request_and_scopes_game(backend_map):
    _, _, calls = backend_map
    assert circuit_sections.get_circuit_sections(telemetry([0.15], game="iracing"))
    assert calls[0] == "circuit-map/list?game=iracing"


def test_unnamed_section_uses_backend_tags(backend_map):
    _, details, _ = backend_map
    details["centerline_segments"][0].pop("name")
    assert circuit_sections.get_circuit_sections(telemetry([0.15]))[0]["name"] == "corner"


def test_missing_map_has_no_hardcoded_fallback(backend_map):
    assert circuit_sections.get_circuit_sections(telemetry([0.15], "brands_hatch")) == []


def test_empty_sections_have_no_fallback(backend_map):
    _, details, _ = backend_map
    details["centerline_segments"] = []
    assert circuit_sections.get_circuit_sections(telemetry([0.15])) == []


def test_backend_failure_has_no_hardcoded_fallback(monkeypatch, backend_map):
    async def fail(*args, **kwargs):
        return {"error": "backend unavailable"}

    monkeypatch.setattr(BackendService, "call_backend_function", fail)
    assert circuit_sections.get_circuit_sections(telemetry([0.15], "brands_hatch")) == []


def test_label_catalog_contains_only_behavior_and_segment_types():
    from app.shared.labels import BEHAVIOR_LABELS, LABEL_CATEGORIES, LABEL_MAPPING
    from app.internal_knowledge_base.label_lookup import find_labels

    assert LABEL_CATEGORIES["Main Labels"] == BEHAVIOR_LABELS
    assert "Track" not in LABEL_CATEGORIES
    assert "brands_hatch3" not in LABEL_MAPPING
    assert LABEL_MAPPING["MSP1"] == "Initiate brake too late"
    assert LABEL_MAPPING["ST1"] == "In the corner"
    assert find_labels(type="circuit_section") == []


def test_display_segments_use_backend_sections_and_keep_label_ranges(backend_map):
    from app.shared.label_hierarchy import build_track_area_segments

    segments = build_track_area_segments(
        [{"id": "prediction", "start_index": 0, "end_index": 4,
          "labels": [{"label_name": "MSP1", "start_index": 1, "end_index": 3}]}],
        telemetry([0.15, 0.25, 0.95, 0.97]).to_dict("records"),
        "custom_layout",
    )
    assert [s["track_section"] for s in segments] == ["backend-a", "backend-b"]
    assert [s["track_section_name"] for s in segments] == ["Backend Hairpin", "Finish Straight"]
    assert [s["labels"] for s in segments] == [
        [{"label_name": "MSP1", "start_index": 1, "end_index": 2}],
        [{"label_name": "MSP1", "start_index": 2, "end_index": 3}],
    ]


def test_classifier_without_track_labels_uses_backend_map(monkeypatch, tmp_path, backend_map):
    from app.api import racing_session
    from app.ml.segment_classifier.service import SegmentClassifierService

    classifier = SegmentClassifierService(str(tmp_path))
    classifier.configure_labels(["MSP1", "ST1"])
    monkeypatch.setattr(classifier, "score_sequence", lambda frame: pd.DataFrame({
        "MSP1": [0.9] * len(frame), "ST1": [0.8] * len(frame),
    }))
    monkeypatch.setattr(racing_session, "get_segment_classifier", lambda: classifier)

    segments = racing_session._classify_telemetry_segments(
        telemetry([0.15, 0.25, 0.95, 0.97]).to_dict("records"), None,
    )

    assert [segment["track_section"] for segment in segments] == ["backend-a", "backend-b"]
    assert [segment["track_section_name"] for segment in segments] == [
        "Backend Hairpin", "Finish Straight",
    ]
    assert [[label["label_name"] for label in segment["labels"]] for segment in segments] == [
        ["MSP1", "ST1"], ["MSP1", "ST1"],
    ]


@pytest.mark.parametrize("position,section_name", [
    (0.2, "Backend Hairpin"), (0.95, "Finish Straight"), (0.05, "Finish Straight"),
])
def test_prompts_resolve_location_from_backend_map(backend_map, position, section_name):
    from app.ml.prompts import generate_llm_prompt_from_labels, generate_training_prompt

    rows = telemetry([position]).to_dict("records")
    for generate in (generate_llm_prompt_from_labels, generate_training_prompt):
        prompt = generate(["EA", "ST1"], telemetry_data=rows)
        assert "custom_layout" in prompt
        assert section_name in prompt
        assert "In the corner" in prompt
        assert "unknown" not in prompt


def test_forecast_only_matches_the_telemetry_track(backend_map):
    from app.ml.opportunity_forecaster.service import match_circuit_section

    rows = telemetry([0.11, 0.15]).to_dict("records")
    rows[0]["Graphics_current_time"] = 0
    rows[1]["Graphics_current_time"] = 1000
    result = match_circuit_section(rows, 2)
    assert result["best_match"]["label_id"] == "backend-a"
    assert result["best_match"]["name"] == "Backend Hairpin"
    assert result["best_match"]["section_range"] == [0.1, 0.3]

    for row in rows:
        row["Static_track"] = "other_track"
    assert match_circuit_section(rows, 2)["top_matches"] == []


@pytest.mark.asyncio
async def test_user_session_summary_uses_backend_section_name(monkeypatch, backend_map):
    from app.services import user_session_analysis
    from app.shared.segment import PredictedSegment

    rows = telemetry([0.15, 0.2, 0.25]).to_dict("records")
    monkeypatch.setattr(
        user_session_analysis, "get_top_lap_reference_model",
        lambda: SimpleNamespace(enrich=lambda records, **kwargs: records),
    )
    monkeypatch.setattr(
        user_session_analysis, "get_segment_classifier",
        lambda: SimpleNamespace(classify_ranges=lambda frame, ranges: [
            PredictedSegment(label="MSP1", score=0.9, start_index=0, end_index=3),
        ]),
    )
    summary = {}
    assert await user_session_analysis._scan_session(
        summary, {"map": "custom_layout", "car_name": "car"}, rows,
    ) == "custom_layout"
    section = summary["tracks"]["custom_layout"]["sections"]["backend-a"]
    assert section["sectionName"] == "Backend Hairpin"
    assert section["labelCounts"] == {"MSP1": 1}
    assert section["mistakes"] == 1
