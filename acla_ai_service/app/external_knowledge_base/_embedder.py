"""Shared embedding model for the racing-engineer KB registry.

One process-wide ``HuggingFaceEmbedding`` instance, consumed by the
LlamaIndex hybrid index in ``_registry.py``. Kept separate from the
annotation embedder so the two registries can configure their models
independently via ``settings.racing_kb_*``.
"""

from __future__ import annotations

import logging
import threading

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

_li_embedding = None
_lock = threading.Lock()


def get_llama_embedding():
    """Return the cached ``HuggingFaceEmbedding``, loading on first call."""
    global _li_embedding
    if _li_embedding is not None:
        return _li_embedding
    with _lock:
        if _li_embedding is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            name = settings.racing_kb_embedding_model
            query_prefix = settings.racing_kb_query_prefix or ""
            LOGGER.info("racing_engineer: loading embedding model '%s' …", name)
            _li_embedding = HuggingFaceEmbedding(
                model_name=name,
                query_instruction=query_prefix,
                text_instruction="",
            )
            LOGGER.info("racing_engineer: embedding model loaded.")
    return _li_embedding


def model_name() -> str:
    """Name of the configured racing-engineer embedding model."""
    return settings.racing_kb_embedding_model


__all__ = ["get_llama_embedding", "model_name"]
