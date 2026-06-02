from __future__ import annotations

from app.pipelines.manifest import node_kinds


def test_opportunity_forecaster_is_model_component_option() -> None:
    training_specs = {
        spec.kind: spec for spec in node_kinds.list_by_category("training")
    }

    assert "opportunity_forecaster" in training_specs
    assert training_specs["opportunity_forecaster"].display == (
        "Opportunity Forecaster Training"
    )
    assert training_specs["opportunity_forecaster"].ui_route == (
        "opportunity_forecaster"
    )
