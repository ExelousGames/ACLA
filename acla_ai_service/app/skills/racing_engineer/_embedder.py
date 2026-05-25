"""Racing-engineer knowledge-base embedding singleton.

Independent of :mod:`app.skills.annotation._embedder` on purpose — the
annotation pipeline uses MiniLM (small, optimised for short labels),
while the racing-engineer corpus runs prose chunks through a stronger
production model (BGE-large by default). Sharing one singleton would
force both to use the same model.

Single process-wide load. Model name comes from
``settings.racing_kb_embedding_model``; calls after the first load
keep the original model regardless of what the setting says — restart
the process to swap models.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Union

import numpy as np

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()
_model_name: str = ""


def _get_model():
    """Return the cached SentenceTransformer, loading on first call."""
    global _model, _model_name
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            name = settings.racing_kb_embedding_model
            LOGGER.info("racing_engineer: loading embedding model '%s' …", name)
            _model = SentenceTransformer(name)
            _model_name = name
            LOGGER.info("racing_engineer: embedding model loaded (%s).", name)
    return _model


def embed_documents(texts: Union[str, List[str]]) -> np.ndarray:
    """Encode corpus chunks. No query prefix — documents go in as-is.

    Always returns a 2-D ``(n, dim)`` array (a single string is wrapped).
    Vectors are L2-normalised so cosine similarity reduces to a dot
    product (the registry does ``M @ q``).
    """
    if isinstance(texts, str):
        texts = [texts]
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]
    return vecs


def embed_query(text: str) -> np.ndarray:
    """Encode a single search query.

    BGE-en-v1.5 was trained with a query-side instruction prefix; the
    prefix is applied automatically per ``settings.racing_kb_query_prefix``.
    Set that setting to an empty string when using a model that doesn't
    want a prefix (e.g. MiniLM, e5-base).
    """
    prefix = settings.racing_kb_query_prefix or ""
    model = _get_model()
    vec = model.encode(
        prefix + text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec


def model_name() -> str:
    """Name of the loaded model — useful for cache invalidation."""
    _get_model()
    return _model_name


__all__ = ["embed_documents", "embed_query", "model_name"]
