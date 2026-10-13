import pytest

from app.ml.opportunity_forecaster.service import OpportunityForecasterService
from training.ml.opportunity_forecaster.trainer import train_opportunity_forecaster


def test_training_artifacts_load_in_runtime_and_preserve_forecasts(tmp_path):
    trained = OpportunityForecasterService(str(tmp_path / "training"))
    rows = [
        {"Physics_speed_kmh": 100.0, "Graphics_gap_ahead": 1.0},
        {"Physics_speed_kmh": 120.0, "Graphics_gap_ahead": 0.5},
    ]
    result = train_opportunity_forecaster(
        [
            {"telemetry_data": rows, "target_label": "O1"},
            {"telemetry_rows": [{"Physics_speed_kmh": 30.0}], "label": "unrelated"},
        ],
        forecaster_service=trained,
    )
    assert result["status"] == "success"
    assert result["samples"] == 2
    assert set(result["classes"]) == {"O1", "NO_OPPORTUNITY"}
    expected = trained.forecast(rows)

    runtime = OpportunityForecasterService(str(tmp_path / "runtime"))
    runtime.deserialize_artifacts(trained.serialize_artifacts())
    assert runtime.load_model()
    assert runtime.feature_names == trained.feature_names
    assert runtime.forecast(rows) == expected


def test_empty_training_data_is_rejected_without_writing_artifacts(tmp_path):
    forecaster = OpportunityForecasterService(str(tmp_path))
    with pytest.raises(ValueError, match="No opportunity forecast training examples"):
        train_opportunity_forecaster([], forecaster_service=forecaster)
    assert not forecaster.has_local_artifacts()
