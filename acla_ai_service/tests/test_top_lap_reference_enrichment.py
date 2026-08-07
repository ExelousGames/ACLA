from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.pipelines.training.pipeline.enrich as enrich_pipeline
from app.api import racing_session
from app.features.tire_grip import TireGripAnalysisService
from app.ml import model_hub
from app.pipelines.inference.preprocessing import InferenceTelemetryBatch
from app.top_laps.runtime import TopLapReferenceModelError
from app.pipelines.training.pipeline.enrich import enrich_sessions_with_context
from app.services import user_session_analysis


@pytest.fixture(autouse=True)
def _runtime_ranges_for_enrichment_tests(monkeypatch):
    monkeypatch.setattr(
        racing_session,
        "split_runtime_segments",
        lambda dataframe, circuit_id: {
            "circuit_id": circuit_id or "brands_hatch",
            "segments": [{"start_index": 0, "end_index": len(dataframe)}],
        },
    )


class EnrichingRuntime:
    def __init__(self, time_differences=None, expert_optimal_times=None):
        self.calls = []
        self.time_differences = time_differences or []
        self.expert_optimal_times = expert_optimal_times or []

    def is_ready(self):
        return True

    def enrich(self, records, track=None, car=None):
        copied = [dict(row) for row in records]
        for index, row in enumerate(copied):
            row["expert_optimal_speed"] = 150.0
            row["expert_time_difference"] = (
                self.time_differences[index]
                if index < len(self.time_differences)
                else 0.0
            )
            row["expert_optimal_time"] = (
                self.expert_optimal_times[index]
                if index < len(self.expert_optimal_times)
                else 90_000.0 + index * 250.0
            )
            row["expert_optimal_player_pos_x"] = 100.0 + index
            row["expert_optimal_player_pos_y"] = 200.0 + index
            row["expert_optimal_player_pos_z"] = 300.0 + index
            row["expert_optimal_throttle"] = round(0.8 - index * 0.1, 1)
            row["expert_optimal_brake"] = round(0.1 + index * 0.1, 1)
            row["expert_optimal_gear"] = 4.0 + index
            if "Static_track" not in row and track:
                row["Static_track"] = track
            if "Static_car_model" not in row and car:
                row["Static_car_model"] = car
        self.calls.append((records, track, car, copied))
        return copied


class UnavailableRuntime:
    def is_ready(self):
        return False

    def enrich(self, records, track=None, car=None):
        raise TopLapReferenceModelError("Top-lap reference model is unavailable")


class EnrichingTireService:
    def __init__(self):
        self.calls = []

    async def enrich(self, records):
        copied = [dict(row) for row in records]
        for index, row in enumerate(copied):
            row["driver_push_to_limit"] = 0.75 + index
        self.calls.append((records, copied))
        return copied


def _preprocessed(records, raw_indices=None):
    return InferenceTelemetryBatch(
        records=[dict(row) for row in records],
        raw_indices=(
            list(raw_indices)
            if raw_indices is not None
            else list(range(len(records)))
        ),
    )


def _tire_grip_row(**overrides):
    row = {
        "Physics_slip_angle_front_left": 0.0,
        "Physics_slip_angle_front_right": 0.0,
        "Physics_slip_angle_rear_left": 0.0,
        "Physics_slip_angle_rear_right": 0.0,
        "Physics_slip_ratio_front_left": 0.0,
        "Physics_slip_ratio_front_right": 0.0,
        "Physics_slip_ratio_rear_left": 0.0,
        "Physics_slip_ratio_rear_right": 0.0,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_tire_grip_service_enriches_aligned_copies():
    source = [
        _tire_grip_row(marker="lateral", Physics_slip_angle_rear_left=0.122),
        _tire_grip_row(marker="longitudinal", Physics_slip_ratio_front_right=0.1),
    ]

    enriched = await TireGripAnalysisService().enrich(source)

    assert enriched is not source
    assert enriched[0] is not source[0]
    assert enriched[0]["marker"] == "lateral"
    assert enriched[0]["driver_push_to_limit"] == pytest.approx(1.0)
    assert enriched[1]["marker"] == "longitudinal"
    assert enriched[1]["driver_push_to_limit"] == pytest.approx(1.0)
    assert all("driver_push_to_limit" not in row for row in source)


@pytest.mark.asyncio
async def test_tire_grip_service_preserves_empty_and_missing_input_behavior():
    service = TireGripAnalysisService()

    assert await service.enrich([]) == []
    with pytest.raises(ValueError, match="missing required columns"):
        await service.enrich([{"Physics_slip_angle_front_left": 0.1}])


@pytest.mark.asyncio
async def test_top_lap_reference_guidance_route_and_handler(monkeypatch):
    calls = []

    async def generate(
        service,
        telemetry_dict,
        *,
        user_request=None,
        track_name=None,
        car_name=None,
    ):
        calls.append(
            (
                service,
                telemetry_dict,
                user_request,
                track_name,
                car_name,
            )
        )
        return {
            "status": "success",
            "timestamp": "2026-01-01T00:00:00",
        }

    monkeypatch.setattr(
        racing_session,
        "generate_top_lap_reference_guidance",
        generate,
    )

    result = await racing_session.get_top_lap_reference_guidance(
        racing_session.TopLapReferenceGuidanceRequest(
            current_telemetry={"Static_track": "spa"},
            human_request="Where can I brake later?",
            track_name="spa",
            car_name="car-a",
        )
    )

    route_paths = {route.path for route in racing_session.router.routes}
    assert "/racing-session/top-lap-reference-guidance" in route_paths
    assert "/racing-session/imitation-learning-guidance" not in route_paths
    assert result["message"] == (
        "Top-lap reference guidance generated successfully"
    )
    assert calls == [
        (
            racing_session.telemetryMLService,
            {"Static_track": "spa"},
            "Where can I brake later?",
            "spa",
            "car-a",
        )
    ]


@pytest.mark.asyncio
async def test_recorded_classifier_receives_enriched_copies(monkeypatch):
    runtime = EnrichingRuntime()
    tire_service = EnrichingTireService()
    source = [
        {"Graphics_normalized_car_position": 0.4, "source": "raw"},
        {"Graphics_normalized_car_position": 0.5, "source": "raw"},
        {"Graphics_normalized_car_position": 0.6, "source": "raw"},
    ]
    cleaned = [{"Graphics_normalized_car_position": 0.6}]
    classified = []
    projected = []
    project_expert_reference_data = (
        racing_session._project_expert_reference_data
    )
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(cleaned, [2]),
    )
    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        racing_session,
        "get_tire_grip_analysis",
        lambda: tire_service,
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        lambda rows, track_name, include_empty_track_sections=False, splitter_result=None: (
            classified.append(rows)
            or [{
                "id": "segment-1",
                "labels": ["EA"],
                "start_index": 0,
                "end_index": 1,
            }]
        ),
    )
    monkeypatch.setattr(
        racing_session,
        "_project_expert_reference_data",
        lambda rows, raw_indices: (
            projected.append(rows)
            or project_expert_reference_data(rows, raw_indices)
        ),
    )

    result = await racing_session.classify_session_segments(
        racing_session.SegmentClassificationRequest(
            session_id="session-1",
            telemetry_data=source,
            track_name="spa",
            car_name="car-a",
        )
    )

    assert result["status"] == "success"
    assert runtime.calls[0][0] == cleaned
    assert classified[0][0]["expert_optimal_speed"] == 150.0
    assert classified[0][0]["driver_push_to_limit"] == 0.75
    assert classified[0][0]["Static_track"] == "spa"
    assert classified[0][0]["Static_car_model"] == "car-a"
    assert result["samples_analyzed"] == 3
    assert result["segments"][0]["start_index"] == 2
    assert result["segments"][0]["end_index"] == 3
    assert result["expert_reference_data"] == [{
        "raw_index": 2,
        "expert_time_difference": 0.0,
        "expert_optimal_time": 90_000.0,
        "expert_optimal_player_pos_x": 100.0,
        "expert_optimal_player_pos_y": 200.0,
        "expert_optimal_player_pos_z": 300.0,
        "Graphics_normalized_car_position": 0.6,
        "expert_optimal_throttle": 0.8,
        "expert_optimal_brake": 0.1,
        "expert_optimal_gear": 4.0,
    }]
    assert tire_service.calls[0][0] is runtime.calls[0][3]
    assert classified[0] is tire_service.calls[0][1]
    assert projected[0] is classified[0]
    assert len(runtime.calls) == 1
    assert source == [
        {"Graphics_normalized_car_position": 0.4, "source": "raw"},
        {"Graphics_normalized_car_position": 0.5, "source": "raw"},
        {"Graphics_normalized_car_position": 0.6, "source": "raw"},
    ]


@pytest.mark.asyncio
async def test_live_gap_uses_the_same_enriched_rows_as_classifier(monkeypatch):
    runtime = EnrichingRuntime(
        time_differences=[10.0, 25.0],
        expert_optimal_times=[91_234.0, 93_456.0],
    )
    tire_service = EnrichingTireService()
    source = [
        {"Graphics_normalized_car_position": 0.0},
        {"Graphics_normalized_car_position": 0.1},
        {"Graphics_normalized_car_position": 0.2},
        {"Graphics_normalized_car_position": 0.3},
    ]
    cleaned = [
        {"Graphics_normalized_car_position": 0.1},
        {"Graphics_normalized_car_position": 0.3},
    ]
    classified = []
    projected = []
    project_expert_reference_data = (
        racing_session._project_expert_reference_data
    )
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(cleaned, [1, 3]),
    )
    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        racing_session,
        "get_tire_grip_analysis",
        lambda: tire_service,
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        lambda rows, track_name, include_empty_track_sections=False, splitter_result=None: (
            classified.append(rows)
            or [{
                "id": "segment-1",
                "labels": ["EA"],
                "start_index": 0,
                "end_index": 2,
            }]
        ),
    )
    monkeypatch.setattr(
        racing_session,
        "_project_expert_reference_data",
        lambda rows, raw_indices: (
            projected.append(rows)
            or project_expert_reference_data(rows, raw_indices)
        ),
    )

    result = await racing_session.analyze_live_baseline(
        racing_session.LiveBaselineAnalysisRequest(
            track="spa",
            car="car-a",
            baseline_lap=2,
            records=source,
        )
    )

    assert result["expert_time_available"] is True
    assert result["segments"][0]["time_gap"] == {
        "start_ms": 10.0,
        "end_ms": 25.0,
        "delta_ms": 15.0,
    }
    assert result["segments"][0]["start_index"] == 1
    assert result["segments"][0]["end_index"] == 4
    assert result["samples_analyzed"] == 4
    assert result["expert_reference_data"] == [
        {
            "raw_index": 1,
            "expert_time_difference": 10.0,
            "expert_optimal_time": 91_234.0,
            "expert_optimal_player_pos_x": 100.0,
            "expert_optimal_player_pos_y": 200.0,
            "expert_optimal_player_pos_z": 300.0,
            "Graphics_normalized_car_position": 0.1,
            "expert_optimal_throttle": 0.8,
            "expert_optimal_brake": 0.1,
            "expert_optimal_gear": 4.0,
        },
        {
            "raw_index": 3,
            "expert_time_difference": 25.0,
            "expert_optimal_time": 93_456.0,
            "expert_optimal_player_pos_x": 101.0,
            "expert_optimal_player_pos_y": 201.0,
            "expert_optimal_player_pos_z": 301.0,
            "Graphics_normalized_car_position": 0.3,
            "expert_optimal_throttle": 0.7,
            "expert_optimal_brake": 0.2,
            "expert_optimal_gear": 5.0,
        },
    ]
    assert runtime.calls[0][0] == cleaned
    assert classified[0][0]["driver_push_to_limit"] == 0.75
    assert tire_service.calls[0][0] is runtime.calls[0][3]
    assert classified[0] is tire_service.calls[0][1]
    assert projected[0] is classified[0]
    assert len(runtime.calls) == 1
    assert source == [
        {"Graphics_normalized_car_position": 0.0},
        {"Graphics_normalized_car_position": 0.1},
        {"Graphics_normalized_car_position": 0.2},
        {"Graphics_normalized_car_position": 0.3},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["recorded", "live"])
async def test_empty_preprocessed_output_returns_empty_expert_references(
    endpoint,
    monkeypatch,
):
    runtime = EnrichingRuntime()
    tire_service = EnrichingTireService()
    source = [{"Graphics_normalized_car_position": 0.5}]
    classified = []

    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed([], []),
    )
    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        racing_session,
        "get_tire_grip_analysis",
        lambda: tire_service,
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        lambda rows, track_name, include_empty_track_sections=False, splitter_result=None: (
            classified.append(rows) or []
        ),
    )

    if endpoint == "recorded":
        result = await racing_session.classify_session_segments(
            racing_session.SegmentClassificationRequest(
                session_id="session-1",
                telemetry_data=source,
                track_name="spa",
                car_name="car-a",
            )
        )
    else:
        result = await racing_session.analyze_live_baseline(
            racing_session.LiveBaselineAnalysisRequest(
                track="spa",
                car="car-a",
                records=source,
            )
        )

    assert result["expert_reference_data"] == []
    assert classified == [[]]
    assert len(runtime.calls) == 1
    assert source == [{"Graphics_normalized_car_position": 0.5}]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["recorded", "live"])
async def test_unavailable_runtime_returns_503_before_classification(
    endpoint,
    monkeypatch,
):
    classifier_called = False

    def classify(*args, **kwargs):
        nonlocal classifier_called
        classifier_called = True
        return []

    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: UnavailableRuntime(),
    )
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(records),
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        classify,
    )

    with pytest.raises(HTTPException) as caught:
        if endpoint == "recorded":
            await racing_session.classify_session_segments(
                racing_session.SegmentClassificationRequest(
                    session_id="session-1",
                    telemetry_data=[{"Graphics_normalized_car_position": 0.5}],
                    track_name="spa",
                    car_name="car-a",
                )
            )
        else:
            await racing_session.analyze_live_baseline(
                racing_session.LiveBaselineAnalysisRequest(
                    track="spa",
                    car="car-a",
                    records=[{"Graphics_normalized_car_position": 0.5}],
                )
            )

    assert caught.value.status_code == 503
    assert classifier_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["recorded", "live"])
async def test_missing_tire_grip_inputs_return_503(
    endpoint,
    monkeypatch,
):
    classifier_called = False

    def classify(*args, **kwargs):
        nonlocal classifier_called
        classifier_called = True
        return []

    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        EnrichingRuntime,
    )
    monkeypatch.setattr(
        racing_session,
        "get_tire_grip_analysis",
        TireGripAnalysisService,
    )
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(records),
    )
    monkeypatch.setattr(
        racing_session,
        "_classify_telemetry_segments",
        classify,
    )

    with pytest.raises(HTTPException) as caught:
        if endpoint == "recorded":
            await racing_session.classify_session_segments(
                racing_session.SegmentClassificationRequest(
                    session_id="session-1",
                    telemetry_data=[{"Graphics_normalized_car_position": 0.5}],
                    track_name="spa",
                    car_name="car-a",
                )
            )
        else:
            await racing_session.analyze_live_baseline(
                racing_session.LiveBaselineAnalysisRequest(
                    track="spa",
                    car="car-a",
                    records=[{"Graphics_normalized_car_position": 0.5}],
                )
            )

    assert caught.value.status_code == 503
    assert "missing required columns" in str(caught.value.detail)
    assert classifier_called is False


@pytest.mark.asyncio
async def test_user_summary_session_classifier_receives_enriched_copies(monkeypatch):
    runtime = EnrichingRuntime()
    tire_service = EnrichingTireService()
    source = [{"Graphics_normalized_car_position": 0.5}]
    classifier_frames = []
    monkeypatch.setattr(
        user_session_analysis,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_tire_grip_analysis",
        lambda: tire_service,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: (
                classifier_frames.append(dataframe) or []
            ),
        ),
    )

    await user_session_analysis._scan_session(
        {"tracks": {}},
        {
            "sessionId": "session-1",
            "map": "brands_hatch",
            "car_name": "car-a",
        },
        source,
    )

    assert classifier_frames[0].iloc[0]["expert_optimal_speed"] == 150.0
    assert classifier_frames[0].iloc[0]["driver_push_to_limit"] == 0.75
    assert classifier_frames[0].iloc[0]["Static_track"] == "brands_hatch"
    assert classifier_frames[0].iloc[0]["Static_car_model"] == "car-a"
    assert tire_service.calls[0][0] is runtime.calls[0][3]
    assert source == [{"Graphics_normalized_car_position": 0.5}]


@pytest.mark.asyncio
async def test_user_summary_runtime_errors_propagate(monkeypatch):
    class Backend:
        async def get_user_analysis_sessions(self, user_id, session_limit):
            return {
                "sessions": [{
                    "sessionId": "session-1",
                    "map": "brands_hatch",
                    "car_name": "car-a",
                }]
            }

        async def iter_user_analysis_chunks(self, user_id, session_meta):
            yield [{"Graphics_normalized_car_position": 0.5}]

    runtime = UnavailableRuntime()
    runtime.is_ready = lambda: True
    monkeypatch.setattr(user_session_analysis, "backend_service", Backend())
    monkeypatch.setattr(
        user_session_analysis,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(records),
    )

    with pytest.raises(TopLapReferenceModelError):
        await user_session_analysis.analyze_user_sessions("user-1")


@pytest.mark.asyncio
async def test_user_summary_tire_grip_errors_mark_session_failed(monkeypatch):
    class Backend:
        async def get_user_analysis_sessions(self, user_id, session_limit):
            return {
                "sessions": [{
                    "sessionId": "session-1",
                    "map": "brands_hatch",
                    "car_name": "car-a",
                }]
            }

        async def iter_user_analysis_chunks(self, user_id, session_meta):
            yield [{"Graphics_normalized_car_position": 0.5}]

    monkeypatch.setattr(user_session_analysis, "backend_service", Backend())
    monkeypatch.setattr(
        user_session_analysis,
        "get_top_lap_reference_model",
        EnrichingRuntime,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_tire_grip_analysis",
        TireGripAnalysisService,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(records),
    )

    result = await user_session_analysis.analyze_user_sessions("user-1")

    assert result["sessionsAnalyzed"] == 0
    assert result["sessionsFailed"] == 1
    assert len(result["errors"]) == 1
    assert "missing required columns" in result["errors"][0]["message"]


@pytest.mark.asyncio
async def test_user_sessions_preprocess_each_assembled_session_once(monkeypatch):
    raw_chunks = [
        [{"raw": 0}, {"raw": 1}],
        [{"raw": 2}, {"raw": 3}],
    ]
    cleaned = [
        {"Graphics_normalized_car_position": 0.1, "clean": 0},
        {"Graphics_normalized_car_position": 0.2, "clean": 1},
        {"Graphics_normalized_car_position": 0.3, "clean": 2},
    ]
    preprocessing_calls = []

    class Backend:
        async def get_user_analysis_sessions(self, user_id, session_limit):
            return {
                "sessions": [{
                    "sessionId": "session-1",
                    "map": "brands_hatch",
                    "car_name": "car-a",
                }]
            }

        async def iter_user_analysis_chunks(self, user_id, session_meta):
            for chunk in raw_chunks:
                yield chunk

    def preprocess(records):
        preprocessing_calls.append(records)
        return _preprocessed(cleaned, [0, 2, 3])

    runtime = EnrichingRuntime()
    classifier_frames = []
    monkeypatch.setattr(user_session_analysis, "backend_service", Backend())
    monkeypatch.setattr(
        user_session_analysis,
        "preprocess_inference_telemetry",
        preprocess,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_top_lap_reference_model",
        lambda: runtime,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_tire_grip_analysis",
        EnrichingTireService,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: (
                classifier_frames.append(dataframe) or []
            ),
        ),
    )

    result = await user_session_analysis.analyze_user_sessions("user-1")

    assert preprocessing_calls == [[
        {"raw": 0},
        {"raw": 1},
        {"raw": 2},
        {"raw": 3},
    ]]
    assert [
        frame["clean"].tolist()
        for frame in classifier_frames
    ] == [[0, 1, 2]]
    assert result["sessionsAnalyzed"] == 1
    assert result["totalTelemetryRows"] == 4
    assert result["tracks"]["brands_hatch"]["totalTelemetryRows"] == 4


@pytest.mark.asyncio
async def test_user_sessions_use_static_track_when_session_map_is_missing(monkeypatch):
    rows = [{
        "Graphics_normalized_car_position": 0.12,
        "Static_track": "brands_hatch",
    }]

    class Backend:
        async def get_user_analysis_sessions(self, user_id, session_limit):
            return {
                "sessions": [{
                    "sessionId": "session-1",
                    "map": None,
                    "car_name": "car-a",
                }]
            }

        async def iter_user_analysis_chunks(self, user_id, session_meta):
            yield rows

    classifier_ranges = []
    monkeypatch.setattr(user_session_analysis, "backend_service", Backend())
    monkeypatch.setattr(
        user_session_analysis,
        "preprocess_inference_telemetry",
        lambda records: _preprocessed(records),
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_top_lap_reference_model",
        EnrichingRuntime,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_tire_grip_analysis",
        EnrichingTireService,
    )
    monkeypatch.setattr(
        user_session_analysis,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: (
                classifier_ranges.append(ranges) or []
            ),
        ),
    )

    result = await user_session_analysis.analyze_user_sessions("user-1")

    assert result["sessionsAnalyzed"] == 1
    assert result["tracks"]["brands_hatch"]["sessionsAnalyzed"] == 1
    assert classifier_ranges[0][0]["circuit_section_id"] == "brands_hatch3"


@pytest.mark.asyncio
async def test_training_enrichment_uses_its_injected_reference_service():
    source = [{"speed": 120}]

    class TrainingReference:
        def __init__(self):
            self.received = None

        def extract_reference_features(self, records):
            self.received = records
            return [{"expert_optimal_speed": 140}]

    class TireService:
        def __init__(self):
            self.received = None

        async def enrich(self, records):
            self.received = records
            return [
                {**record, "estimated_tire_grip": 0.95}
                for record in records
            ]

    reference = TrainingReference()
    tire_service = TireService()
    enriched = await enrich_sessions_with_context(
        source,
        reference,
        tire_service,
    )

    assert reference.received is source
    assert tire_service.received[0]["expert_optimal_speed"] == 140
    assert enriched == [{
        "speed": 120,
        "expert_optimal_speed": 140,
        "estimated_tire_grip": 0.95,
    }]
    assert source == [{"speed": 120}]


@pytest.mark.asyncio
async def test_pipeline_builds_uploads_and_reuses_local_reference(
    monkeypatch,
):
    events = []

    class LocalReference:
        async def build_from_cached_top_laps(self, cache_key):
            events.append(("build", cache_key))
            return {"reference_summary": {"buckets_recorded": 1}}

        def serialize_reference_model(self):
            events.append(("serialize",))
            return {"top_lap_store": {"spa|car-a|grip2": "encoded"}}

        def extract_reference_features(self, records):
            events.append(("extract", records))
            return [{"expert_optimal_speed": 150.0} for _ in records]

    local_reference = LocalReference()
    monkeypatch.setattr(
        enrich_pipeline,
        "TopLapReferenceModelService",
        lambda **_kwargs: local_reference,
    )

    class TireService:
        feature_catalog = SimpleNamespace(
            CONTEXT_FEATURES=["estimated_tire_grip"]
        )

        async def train_tire_grip_model_streaming(self, chunk_iterator):
            list(chunk_iterator)
            return {"success": True}

        def serialize_tire_grip_model(self):
            return {"serialized_timestamp": "2026-01-01T00:00:00"}

        async def enrich(self, records):
            return [
                {**record, "estimated_tire_grip": 0.95}
                for record in records
            ]

    monkeypatch.setattr(
        enrich_pipeline,
        "TireGripAnalysisService",
        TireService,
    )

    def reject_model_hub_access():
        raise AssertionError("pipeline accessed the model-hub reference")

    monkeypatch.setattr(
        model_hub,
        "get_top_lap_reference_model",
        reject_model_hub_access,
    )

    class TelemetryStore:
        def __init__(self):
            self.cached = []

        def has_cached_data(self, cache_key):
            return cache_key == "top-laps"

        def get_cached_data_chunks(self, cache_key, include_ids=False):
            if cache_key == "top-laps":
                return iter([[[{"Static_track": "spa"}]]])
            rows = [{"Graphics_normalized_car_position": 0.5}]
            if include_ids:
                return iter([(rows, "session-1")])
            return iter([rows])

        async def cache_chunks_streaming(self, cache_key, chunks_iterator):
            async for payload in chunks_iterator:
                self.cached.append((cache_key, payload))
            return True

    class Backend:
        def __init__(self):
            self.saved = []

        async def save_ai_model(self, **kwargs):
            self.saved.append(kwargs)

    telemetry_store = TelemetryStore()
    backend = Backend()
    cache_config = SimpleNamespace(
        top_laps_cache_key="top-laps",
        enriched_sessions_cache_key="enriched-sessions",
    )

    cache_key, returned_reference = (
        await enrich_pipeline.enriched_contextual_data(
            "sessions",
            telemetry_store=telemetry_store,
            cache_config=cache_config,
            backend_service=backend,
        )
    )

    assert cache_key == "enriched-sessions"
    assert returned_reference is local_reference
    assert events[0:2] == [("build", "top-laps"), ("serialize",)]
    assert any(event[0] == "extract" for event in events)
    assert backend.saved[0] == {
        "model_type": "top_lap_reference",
        "model_data": {
            "top_lap_store": {"spa|car-a|grip2": "encoded"}
        },
        "metadata": {"buckets_recorded": 1},
        "is_active": True,
    }
    enriched_payload = telemetry_store.cached[0][1][0]
    assert enriched_payload[0]["expert_optimal_speed"] == 150.0
