"""Hydrate the opportunity forecaster from backend model storage on startup."""

from __future__ import annotations

import logging

from app.integrations.backend.client import backend_service
from app.ml.opportunity_forecaster.service import opportunity_forecaster

LOGGER = logging.getLogger(__name__)

_MODEL_TYPE = "opportunity_forecaster"


async def ensure_opportunity_forecaster_model() -> bool:
    if opportunity_forecaster.has_local_artifacts():
        LOGGER.info(
            "opportunity_forecaster artifacts already present at %s",
            opportunity_forecaster.models_directory,
        )
        return True

    LOGGER.info("opportunity_forecaster artifacts missing - fetching from backend (%s)", _MODEL_TYPE)
    try:
        active = await backend_service.getCompleteActiveModelData(modelType=_MODEL_TYPE)
    except Exception as exc:
        LOGGER.warning(
            "opportunity_forecaster backend fetch failed: %s - forecast will report not_trained until local train.",
            exc,
        )
        return False

    try:
        opportunity_forecaster.deserialize_artifacts(active.modelData)
    except Exception as exc:
        LOGGER.warning("opportunity_forecaster payload could not be written to disk: %s", exc)
        return False

    LOGGER.info("opportunity_forecaster artifacts hydrated into %s", opportunity_forecaster.models_directory)
    return True


__all__ = ["ensure_opportunity_forecaster_model"]
