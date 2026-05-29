"""Hydrate the segment classifier from the backend on FastAPI startup.

Mirrors :func:`app.llama.chat_model.ensure_chat_gguf`: if the local model files
exist we no-op; otherwise we fetch the active payload from the backend and
write the artifacts back into ``segment_classifier.models_directory``.

Backend upload happens at the tail of
:meth:`SegmentClassifierService.train_model`, so the round-trip is
train → backend → next-boot → disk → lazy ``load_model()``.
"""

from __future__ import annotations

import logging

from app.integrations.backend.client import backend_service
from app.ml.segment_classifier.service import segment_classifier

LOGGER = logging.getLogger(__name__)

_MODEL_TYPE = "segment_classifier"


async def ensure_segment_classifier_model() -> bool:
    """Return True if artifacts are on disk after this call, False otherwise."""
    if segment_classifier.has_local_artifacts():
        LOGGER.info("segment_classifier artifacts already present at %s", segment_classifier.models_directory)
        return True

    LOGGER.info("segment_classifier artifacts missing — fetching from backend (%s)", _MODEL_TYPE)
    try:
        active = await backend_service.getCompleteActiveModelData(modelType=_MODEL_TYPE)
    except Exception as exc:
        LOGGER.warning(
            "segment_classifier backend fetch failed: %s — classifier will report 'not trained' until a local train.",
            exc,
        )
        return False

    try:
        segment_classifier.deserialize_artifacts(active.modelData)
    except Exception as exc:
        LOGGER.warning("segment_classifier payload could not be written to disk: %s", exc)
        return False

    LOGGER.info("segment_classifier artifacts hydrated into %s", segment_classifier.models_directory)
    return True


__all__ = ["ensure_segment_classifier_model"]
