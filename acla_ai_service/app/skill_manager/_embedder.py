"""Shared SentenceTransformer singleton.

One process-wide model load, reused by the SkillRegistry's discovery
index and by any sub-agent that needs to embed text (e.g. label_verifier).
"""

from __future__ import annotations

import logging
import threading
from typing import List, Union

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_model = None
_lock = threading.Lock()
_model_name: str = DEFAULT_MODEL


def get_model(model_name: str = DEFAULT_MODEL):
    """Return the cached SentenceTransformer, loading on first call.

    Subsequent calls with a different model_name keep the original model
    (the singleton is process-wide). To switch, restart the process.
    """
    global _model, _model_name
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            LOGGER.info("Loading embedding model '%s' …", model_name)
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            LOGGER.info("Embedding model loaded.")
    return _model


def embed(text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    """Encode text to a numpy vector (or 2-D array for a list).

    Normalised by default so cosine reduces to a dot product.
    """
    model = get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Pure cosine for unnormalised vectors. For normalised inputs, use `a @ b`."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
