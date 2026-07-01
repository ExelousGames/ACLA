"""Shared embedding model for annotation label retrieval.

Hosts one process-wide ``HuggingFaceEmbedding`` instance, consumed by
the hybrid label index in :mod:`app.internal_knowledge_base.label_search`.

To swap the model, change ``settings.annotation_skill_embedding_model``
and restart — the singleton is process-wide.
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
            name = settings.annotation_skill_embedding_model
            query_prefix = settings.annotation_skill_query_prefix or ""
            LOGGER.info("annotation: loading embedding model '%s' …", name)
            _li_embedding = HuggingFaceEmbedding(
                model_name=name,
                query_instruction=query_prefix,
                text_instruction="",
            )
            LOGGER.info("annotation: embedding model loaded.")
    return _li_embedding


__all__ = ["get_llama_embedding"]
