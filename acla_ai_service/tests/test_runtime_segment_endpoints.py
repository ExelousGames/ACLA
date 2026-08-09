from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import racing_session
from app.pipelines.inference.preprocessing import InferenceTelemetryBatch


def _configure_analysis_dependencies(monkeypatch):
    monkeypatch.setattr(
        racing_session,
        "preprocess_inference_telemetry",
        lambda records: InferenceTelemetryBatch(
            records=[dict(row) for row in records],
            raw_indices=list(range(len(records))),
        ),
    )
    monkeypatch.setattr(
        racing_session,
        "get_top_lap_reference_model",
        lambda: SimpleNamespace(
            enrich=lambda records, track=None, car=None: records,
        ),
    )

    class TireGrip:
        async def enrich(self, records):
            return records

    monkeypatch.setattr(racing_session, "get_tire_grip_analysis", TireGrip)
    monkeypatch.setattr(
        racing_session,
        "_project_expert_reference_data",
        lambda enriched_rows, raw_indices: [],
    )
    monkeypatch.setattr(
        racing_session,
        "get_segment_classifier",
        lambda: SimpleNamespace(classify_ranges=lambda dataframe, ranges: []),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["segment", "live"])
@pytest.mark.parametrize(
    ("records", "track"),
    [
        ([{"Static_track": "brands_hatch"}], "brands_hatch"),
        ([{"Graphics_normalized_car_position": 0.12}], None),
    ],
)
async def test_direct_analysis_returns_422_for_unsplittable_telemetry(
    endpoint,
    records,
    track,
    monkeypatch,
):
    _configure_analysis_dependencies(monkeypatch)

    with pytest.raises(HTTPException) as caught:
        if endpoint == "segment":
            await racing_session.classify_session_segments(
                racing_session.SegmentClassificationRequest(
                    telemetry_data=records,
                    track_name=track,
                )
            )
        else:
            await racing_session.analyze_live_baseline(
                racing_session.LiveBaselineAnalysisRequest(
                    records=records,
                    track=track,
                )
            )

    assert caught.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["segment", "live"])
async def test_following_only_direct_analysis_succeeds_with_zero_segments(
    endpoint,
    monkeypatch,
):
    _configure_analysis_dependencies(monkeypatch)
    monkeypatch.setattr(
        racing_session,
        "split_runtime_segments",
        lambda dataframe, circuit_id: {
            "circuit_id": "brands_hatch",
            "opponent_session": True,
            "segments": [],
        },
    )
    classified_ranges = []
    monkeypatch.setattr(
        racing_session,
        "get_segment_classifier",
        lambda: SimpleNamespace(
            classify_ranges=lambda dataframe, ranges: (
                classified_ranges.append(ranges) or []
            ),
        ),
    )
    records = [{"Graphics_normalized_car_position": 0.12}]

    if endpoint == "segment":
        result = await racing_session.classify_session_segments(
            racing_session.SegmentClassificationRequest(
                telemetry_data=records,
                track_name="brands_hatch",
            )
        )
    else:
        result = await racing_session.analyze_live_baseline(
            racing_session.LiveBaselineAnalysisRequest(
                records=records,
                track="brands_hatch",
            )
        )

    assert result["status"] == "success"
    assert result["parent_segment_count"] == 0
    assert result["segments"] == []
    assert "expert_reference_data" not in result
    assert classified_ranges == [[]]


@pytest.mark.asyncio
async def test_user_analysis_endpoint_has_no_cropper_readiness_gate(monkeypatch):
    async def analyze(user_id, session_limit):
        return {"userId": user_id, "sessionLimit": session_limit}

    monkeypatch.setattr(racing_session, "analyze_user_sessions", analyze)

    result = await racing_session.analyze_all_user_sessions(
        racing_session.AnalyzeUserSessionsRequest(
            user_id="user-1",
            session_limit=3,
        )
    )

    assert not hasattr(racing_session, "_require_segment_cropper")
    assert result["sessionAnalysis"] == {"userId": "user-1", "sessionLimit": 3}
