"""Fit opportunity models using the shared inference feature contract."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.ml.opportunity_forecaster.service import (
    FORECAST_LABELS,
    NO_OPPORTUNITY,
    OpportunityForecasterService,
    opportunity_forecaster,
)


def train_opportunity_forecaster(
    examples: Iterable[Dict[str, Any]],
    *,
    forecaster_service: OpportunityForecasterService = opportunity_forecaster,
) -> Dict[str, Any]:
    feature_rows: List[Dict[str, float]] = []
    labels: List[str] = []
    for example in examples:
        rows = example.get("telemetry_data") or example.get("telemetry_rows") or []
        label = str(example.get("label") or example.get("target_label") or NO_OPPORTUNITY)
        if label not in FORECAST_LABELS:
            label = NO_OPPORTUNITY
        feature_rows.append(forecaster_service.extract_features(rows))
        labels.append(label)

    if not feature_rows:
        raise ValueError("No opportunity forecast training examples provided")

    forecaster_service.feature_names = sorted({name for row in feature_rows for name in row})
    x = np.asarray(
        [[float(row.get(name, 0.0)) for name in forecaster_service.feature_names] for row in feature_rows],
        dtype=float,
    )
    forecaster_service.scaler = StandardScaler()
    x_scaled = forecaster_service.scaler.fit_transform(x)
    forecaster_service.model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
    )
    forecaster_service.model.fit(x_scaled, labels)
    forecaster_service.save_artifacts()
    return {
        "status": "success",
        "samples": len(labels),
        "classes": list(forecaster_service.model.classes_),
        "feature_count": len(forecaster_service.feature_names),
    }

