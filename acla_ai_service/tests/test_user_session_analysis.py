from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import user_session_analysis as analysis


class FakeBackend:
    async def get_user_analysis_sessions(self, user_id):
        return {
            "sessions": [
                {
                    "sessionId": "brands-session",
                    "map": "Brands Hatch",
                    "car_name": "BMW",
                    "totalChunks": 1,
                },
                {
                    "sessionId": "silverstone-session",
                    "map": "Silverstone",
                    "car_name": "BMW",
                    "totalChunks": 1,
                },
            ]
        }

    async def iter_user_analysis_chunks(self, user_id, session_meta):
        if session_meta["sessionId"] == "brands-session":
            yield [
                {"Graphics_normalized_car_position": 0.12},
                {"Graphics_normalized_car_position": 0.13},
                {"Graphics_normalized_car_position": 0.14},
                {"Graphics_normalized_car_position": 0.15},
            ]
        else:
            yield [{"Graphics_normalized_car_position": 0.1}]


class FakeClassifier:
    def scan_telemetry_data(self, dataframe):
        return [
            SimpleNamespace(start_index=0, end_index=2, labels=["EA"]),
            SimpleNamespace(start_index=2, end_index=4, labels=["MSP1"]),
        ]


class FailingClassifier:
    def scan_telemetry_data(self, dataframe):
        raise RuntimeError("classifier unavailable")


@pytest.mark.asyncio
async def test_analyze_user_sessions_aggregates_measured_sections(monkeypatch):
    monkeypatch.setattr(analysis, "backend_service", FakeBackend())
    monkeypatch.setattr(analysis, "segment_classifier", FakeClassifier())

    result = await analysis.analyze_user_sessions("user-1")

    assert result["sessionsAnalyzed"] == 1
    assert result["sessionsSkipped"] == 1
    assert result["sessionsFailed"] == 0
    assert result["totalTelemetryRows"] == 4
    druids = result["tracks"]["brands_hatch"]["sections"]["brands_hatch3"]
    assert druids["sectionName"] == "Druids"
    assert druids["expertLevelTurns"] == 1
    assert druids["mistakes"] == 1
    assert druids["practiceMistakes"] == 1
    assert druids["labelCounts"] == {"EA": 1, "MSP1": 1}


def test_section_for_rows_uses_measured_normalized_position():
    section_id = analysis._section_for_rows(
        [{"Graphics_normalized_car_position": 0.12}],
        "brands_hatch",
    )

    assert section_id == "brands_hatch3"


@pytest.mark.asyncio
async def test_analyze_user_sessions_reports_classifier_failures(monkeypatch):
    monkeypatch.setattr(analysis, "backend_service", FakeBackend())
    monkeypatch.setattr(analysis, "segment_classifier", FailingClassifier())

    result = await analysis.analyze_user_sessions("user-1")

    assert result["sessionsAnalyzed"] == 0
    assert result["sessionsSkipped"] == 1
    assert result["sessionsFailed"] == 1
    assert result["errors"] == [
        {
            "sessionId": "brands-session",
            "trackId": "brands_hatch",
            "message": "classifier unavailable",
        }
    ]
    assert result["tracks"]["brands_hatch"]["sessionsFailed"] == 1
