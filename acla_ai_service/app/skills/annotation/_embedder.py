"""Shared embedding model for the annotation skill registry.

Hosts one ``HuggingFaceEmbedding`` instance — consumed by the LlamaIndex
hybrid index in ``_registry.py`` and reused by sub-agents (notably
``label_verifier``) that score observations against label descriptions
via the legacy ``embed`` / ``cosine_sim`` numpy helpers. Both routes
share the same underlying SentenceTransformer load.

To swap the model, change ``settings.annotation_skill_embedding_model``
and restart — the singleton is process-wide.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Union

import numpy as np

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


def model_name() -> str:
    """Name of the configured annotation embedding model."""
    return settings.annotation_skill_embedding_model


def embed(text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    """Encode text via the shared SentenceTransformer.

    Used by sub-agents (e.g. ``label_verifier``) for ad-hoc similarity
    scoring outside the registry's hybrid index. Returns a 1-D vector
    for a str input or a 2-D ``(n, dim)`` array for a list input.
    Normalised by default so cosine reduces to a dot product.
    """
    st_model = get_llama_embedding()._model
    return st_model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine for unnormalised vectors. For normalised inputs, use ``a @ b``."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


__all__ = ["cosine_sim", "embed", "get_llama_embedding", "model_name"]
