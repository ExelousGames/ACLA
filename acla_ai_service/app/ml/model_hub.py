"""Central runtime model hub for chatbot-facing models.

The backend active model store is the source of truth at process startup.
Each model keeps its own existing serialization format; this module only
coordinates download, hydration, readiness, and shared access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.integrations.backend.client import backend_service

LOGGER = logging.getLogger(__name__)

ModelPayload = Dict[str, Any]
Hydrator = Callable[[ModelPayload], bool]
ReadyCheck = Callable[[], bool]


@dataclass(frozen=True)
class ChatbotModelSpec:
    """Backend model registration for one chatbot runtime dependency."""

    name: str
    backend_model_type: str
    hydrate: Hydrator
    is_ready: ReadyCheck


_hydration_status: Dict[str, bool] = {}


def get_segment_classifier():
    from app.ml.segment_classifier.service import segment_classifier

    return segment_classifier


def get_opportunity_forecaster():
    from app.ml.opportunity_forecaster.service import opportunity_forecaster

    return opportunity_forecaster


def get_top_lap_reference_model():
    from app.top_laps.runtime import top_lap_reference_model

    return top_lap_reference_model


def get_tire_grip_analysis():
    from app.features.tire_grip import tire_grip_analysis_service

    return tire_grip_analysis_service


def _segment_classifier_ready() -> bool:
    segment_classifier = get_segment_classifier()
    return (
        segment_classifier.model is not None
        and segment_classifier.scaler is not None
        and bool(segment_classifier.label_ids)
    )


def _opportunity_forecaster_ready() -> bool:
    opportunity_forecaster = get_opportunity_forecaster()
    return opportunity_forecaster.model is not None and opportunity_forecaster.scaler is not None


def _top_lap_reference_ready() -> bool:
    return get_top_lap_reference_model().is_ready()


def _tire_grip_ready() -> bool:
    return _hydration_status.get("tire_grip_analysis", False)


def _hydrate_segment_classifier(payload: ModelPayload) -> bool:
    segment_classifier = get_segment_classifier()
    segment_classifier.deserialize_artifacts(payload)
    return bool(segment_classifier.load_model())


def _hydrate_opportunity_forecaster(payload: ModelPayload) -> bool:
    opportunity_forecaster = get_opportunity_forecaster()
    opportunity_forecaster.deserialize_artifacts(payload)
    return bool(opportunity_forecaster.load_model())


def _hydrate_top_lap_reference(payload: ModelPayload) -> bool:
    get_top_lap_reference_model().install_backend_payload(payload)
    return _top_lap_reference_ready()


def _hydrate_tire_grip(payload: ModelPayload) -> bool:
    tire_grip_analysis_service = get_tire_grip_analysis()
    tire_grip_analysis_service.deserialize_tire_grip_model(payload)
    return True


_MODEL_SPECS = (
    ChatbotModelSpec(
        name="segment_classifier",
        backend_model_type="segment_classifier",
        hydrate=_hydrate_segment_classifier,
        is_ready=_segment_classifier_ready,
    ),
    ChatbotModelSpec(
        name="opportunity_forecaster",
        backend_model_type="opportunity_forecaster",
        hydrate=_hydrate_opportunity_forecaster,
        is_ready=_opportunity_forecaster_ready,
    ),
    ChatbotModelSpec(
        name="top_lap_reference",
        backend_model_type="top_lap_reference",
        hydrate=_hydrate_top_lap_reference,
        is_ready=_top_lap_reference_ready,
    ),
    ChatbotModelSpec(
        name="tire_grip_analysis",
        backend_model_type="tire_grip_analysis",
        hydrate=_hydrate_tire_grip,
        is_ready=_tire_grip_ready,
    ),
)


async def _hydrate_model(spec: ChatbotModelSpec, backend: Any) -> bool:
    try:
        is_ready = spec.is_ready()
    except Exception as exc:  # noqa: BLE001 - missing optional deps are per-model degraded readiness
        is_ready = False
        LOGGER.warning("%s readiness check failed: %s", spec.name, exc)

    if is_ready:
        _hydration_status[spec.name] = True
        LOGGER.info("%s already ready; skipping backend download", spec.name)
        return True

    try:
        active = await backend.getCompleteActiveModelData(modelType=spec.backend_model_type)
    except Exception as exc:  # noqa: BLE001 - one failed model must not stop startup
        _hydration_status[spec.name] = False
        LOGGER.warning("%s backend download failed: %s", spec.name, exc)
        return False

    try:
        ok = bool(spec.hydrate(active.modelData))
    except Exception as exc:  # noqa: BLE001 - keep startup resilient per model
        _hydration_status[spec.name] = False
        LOGGER.warning("%s payload hydration failed: %s", spec.name, exc)
        return False

    _hydration_status[spec.name] = ok
    if ok:
        LOGGER.info("%s hydrated from backend active model store", spec.name)
    else:
        LOGGER.warning("%s backend payload hydrated but model is not ready", spec.name)
    return ok


async def hydrate_chatbot_models(backend: Optional[Any] = None) -> Dict[str, bool]:
    """Download and hydrate all chatbot-facing models from backend storage."""

    backend_client = backend or backend_service
    # Runtime top-lap reference data is backend-owned. Clear readiness before
    # every startup hydration and never reconstruct it from a local artifact.
    get_top_lap_reference_model().reset()
    _hydration_status["top_lap_reference"] = False
    results: Dict[str, bool] = {}
    for spec in _MODEL_SPECS:
        results[spec.name] = await _hydrate_model(spec, backend_client)
    return results


def get_chatbot_model_status() -> Dict[str, bool]:
    """Return the latest hub readiness snapshot."""

    status: Dict[str, bool] = {}
    for spec in _MODEL_SPECS:
        try:
            status[spec.name] = bool(_hydration_status.get(spec.name) or spec.is_ready())
        except Exception:
            status[spec.name] = bool(_hydration_status.get(spec.name, False))
    return status


__all__ = [
    "get_chatbot_model_status",
    "get_opportunity_forecaster",
    "get_top_lap_reference_model",
    "get_segment_classifier",
    "get_tire_grip_analysis",
    "hydrate_chatbot_models",
]
