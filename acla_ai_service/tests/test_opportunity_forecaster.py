from __future__ import annotations

import numpy as np

from app.ml.opportunity_forecaster.service import (
    NO_OPPORTUNITY,
    OpportunityForecasterService,
    match_circuit_section,
)


def test_feature_extraction_handles_missing_optional_columns() -> None:
    service = OpportunityForecasterService()

    features = service.extract_features([
        {
            "Graphics_normalized_car_position": 0.5,
            "Graphics_current_time": 1000,
            "Physics_speed_kmh": 120,
        }
    ])

    assert features["sample_count"] == 1.0
    assert features["Physics_speed_kmh_last"] == 120.0
    assert features["Graphics_gap_ahead_mean"] == 0.0
    assert features["nearest_opponent_min_distance_m"] == 0.0


def test_circuit_section_projection_identifies_sheene_curve() -> None:
    rows = [
        {"Graphics_normalized_car_position": 0.66, "Graphics_current_time": 0},
        {"Graphics_normalized_car_position": 0.67, "Graphics_current_time": 1000},
    ]

    result = match_circuit_section(rows, horizon_seconds=5)

    assert result["best_match"]["label_id"] == "brands_hatch13"
    assert result["best_match"]["name"] == "Sheene Curve"


class _FakeModel:
    classes_ = np.array([NO_OPPORTUNITY, "O4", "OD1"])

    def predict_proba(self, _x):
        return np.array([[0.2, 0.7, 0.1]])


class _FakeScaler:
    def transform(self, x):
        return x


def test_forecast_sorts_probabilities_and_excludes_no_opportunity() -> None:
    service = OpportunityForecasterService()
    service.model = _FakeModel()
    service.scaler = _FakeScaler()
    service.feature_names = sorted(service.extract_features([]))

    result = service.forecast(
        [
            {"Graphics_normalized_car_position": 0.66, "Graphics_current_time": 0},
            {"Graphics_normalized_car_position": 0.67, "Graphics_current_time": 1000},
        ],
        horizon_seconds=5,
        top_k=2,
    )

    assert result["status"] == "success"
    assert [item["label_id"] for item in result["opportunities"]] == ["O4", "OD1"]
    assert result["opportunities"][0]["probability"] == 0.7
