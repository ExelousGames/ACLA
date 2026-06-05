from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import racing_session


@dataclass
class FakeSegment:
    id: str
    labels: list[str]
    start_index: int
    end_index: int
    telemetry_data: list[dict]

    def to_dict(self):
        return {
            "id": self.id,
            "labels": self.labels,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "telemetry_data": self.telemetry_data,
        }


class FakeClassifier:
    def scan_telemetry_data(self, dataframe):
        return [
            FakeSegment(
                id="segment-1",
                labels=["EA"],
                start_index=0,
                end_index=2,
                telemetry_data=dataframe.iloc[0:2].to_dict("records"),
            )
        ]


class MissingModelClassifier:
    def scan_telemetry_data(self, dataframe):
        raise ValueError("Segment classifier model not trained or found.")


class MainFirstClassifier:
    def scan_telemetry_data(self, dataframe):
        return [
            FakeSegment(
                id="segment-1",
                labels=["MSP1"],
                start_index=0,
                end_index=2,
                telemetry_data=dataframe.iloc[0:2].to_dict("records"),
            ),
            FakeSegment(
                id="segment-2",
                labels=["MSP2", "ST3"],
                start_index=2,
                end_index=4,
                telemetry_data=dataframe.iloc[2:4].to_dict("records"),
            ),
            FakeSegment(
                id="segment-3",
                labels=["RM1"],
                start_index=4,
                end_index=5,
                telemetry_data=dataframe.iloc[4:5].to_dict("records"),
            ),
        ]


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(racing_session.router)
    return TestClient(app)


def test_segment_classification_returns_compact_segments(monkeypatch):
    monkeypatch.setattr(racing_session, "segment_classifier", FakeClassifier())
    client = make_client()

    response = client.post(
        "/racing-session/segment-classification",
        json={
            "session_id": "session-1",
            "telemetry_data": [{"speed": 100}, {"speed": 120}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "session_id": "session-1",
        "samples_analyzed": 2,
        "segment_count": 1,
        "segments": [
            {
                "id": "segment-1",
                "labels": ["EA"],
                "main_label_id": "EA",
                "main_label_name": "Expert Adherence (Training)",
                "start_index": 0,
                "end_index": 2,
                "sub_labels": [],
                "sub_segments": [
                    {
                        "start_index": 0,
                        "end_index": 2,
                        "labels": [],
                    }
                ],
            }
        ],
    }


def test_segment_classification_splits_by_main_label_first(monkeypatch):
    monkeypatch.setattr(racing_session, "segment_classifier", MainFirstClassifier())
    client = make_client()

    response = client.post(
        "/racing-session/segment-classification",
        json={
            "session_id": "session-1",
            "telemetry_data": [
                {"speed": 100},
                {"speed": 110},
                {"speed": 120},
                {"speed": 130},
                {"speed": 90},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["segment_count"] == 2
    assert body["segments"][0] == {
        "id": "segment-1",
        "labels": ["MSP", "MSP1", "MSP2", "ST3"],
        "main_label_id": "MSP",
        "main_label_name": "Mistake (Practice)",
        "start_index": 0,
        "end_index": 4,
        "sub_labels": [
            {"label_id": "MSP1", "label_name": "Initiate brake too late"},
            {"label_id": "MSP2", "label_name": "Initiate the turn too late"},
            {"label_id": "ST3", "label_name": "Approach to corner"},
        ],
        "sub_segments": [
            {
                "start_index": 0,
                "end_index": 2,
                "labels": [
                    {"label_id": "MSP1", "label_name": "Initiate brake too late"},
                ],
            },
            {
                "start_index": 2,
                "end_index": 4,
                "labels": [
                    {"label_id": "MSP2", "label_name": "Initiate the turn too late"},
                    {"label_id": "ST3", "label_name": "Approach to corner"},
                ],
            },
        ],
    }
    assert body["segments"][1]["main_label_id"] == "RM"


def test_segment_classification_rejects_empty_telemetry(monkeypatch):
    monkeypatch.setattr(racing_session, "segment_classifier", FakeClassifier())
    client = make_client()

    response = client.post(
        "/racing-session/segment-classification",
        json={"session_id": "session-1", "telemetry_data": []},
    )

    assert response.status_code == 400


def test_segment_classification_surfaces_missing_model(monkeypatch):
    monkeypatch.setattr(racing_session, "segment_classifier", MissingModelClassifier())
    client = make_client()

    response = client.post(
        "/racing-session/segment-classification",
        json={"session_id": "session-1", "telemetry_data": [{"speed": 100}]},
    )

    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]
