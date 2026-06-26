"""Cross-encoder reranking for annotation label candidates."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from app.infra.config import settings

LOGGER = logging.getLogger(__name__)

_lock = threading.Lock()
_reranker = None
_reranker_model_name: Optional[str] = None
_failed_model_name: Optional[str] = None


def rerank_label_docs(
    query: str,
    docs: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return label docs ordered by semantic fit to the evidence text.

    The embedding score remains available as ``embedding_score``. When the
    reranker is enabled and available, ``score`` becomes the reranker score and
    ``reranker_score`` is added for diagnostics. If the reranker is disabled or
    cannot load, this falls back to the original embedding order.
    """
    candidates = [_with_embedding_score(doc) for doc in docs]
    limit = _top_k(top_k)
    if not candidates:
        return []

    if not settings.annotation_label_reranker_enabled:
        return _embedding_sorted(candidates)[:limit]

    model = _get_cross_encoder()
    if model is None:
        return _embedding_sorted(candidates)[:limit]

    evidence = (query or "").strip()
    if not evidence:
        return _embedding_sorted(candidates)[:limit]

    pairs = [[evidence, _label_doc_text(doc)] for doc in candidates]
    try:
        raw_scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "annotation label reranker failed during prediction; "
            "falling back to embedding scores: %s",
            exc,
        )
        return _embedding_sorted(candidates)[:limit]

    scored: List[Dict[str, Any]] = []
    threshold = settings.annotation_label_reranker_min_score
    if min_score is not None:
        threshold = min_score
    for doc, raw_score in zip(candidates, raw_scores):
        try:
            reranker_score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if threshold is not None and reranker_score < float(threshold):
            LOGGER.debug(
                "annotation label reranker rejected %s: score %.4f < %.4f",
                doc.get("id"),
                reranker_score,
                float(threshold),
            )
            continue
        scored.append({
            **doc,
            "score": reranker_score,
            "reranker_score": reranker_score,
        })

    return sorted(
        scored,
        key=lambda item: float(item.get("reranker_score", item.get("score", 0.0))),
        reverse=True,
    )[:limit]


def _get_cross_encoder():
    global _failed_model_name, _reranker, _reranker_model_name

    model_name = settings.annotation_label_reranker_model
    if not model_name:
        return None
    if _reranker is not None and _reranker_model_name == model_name:
        return _reranker
    if _failed_model_name == model_name:
        return None

    with _lock:
        if _reranker is not None and _reranker_model_name == model_name:
            return _reranker
        if _failed_model_name == model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError as exc:
                LOGGER.warning(
                    "sentence-transformers CrossEncoder is unavailable; "
                    "annotation label reranking is disabled: %s",
                    exc,
                )
                _failed_model_name = model_name
                return None
        try:
            LOGGER.info("annotation: loading label reranker '%s'.", model_name)
            _reranker = CrossEncoder(model_name)
            _reranker_model_name = model_name
            _failed_model_name = None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "annotation label reranker '%s' could not be loaded; "
                "falling back to embedding scores: %s",
                model_name,
                exc,
            )
            _failed_model_name = model_name
            return None
    return _reranker


def _with_embedding_score(doc: Dict[str, Any]) -> Dict[str, Any]:
    score = float(doc.get("embedding_score", doc.get("score", 0.0)) or 0.0)
    return {**doc, "embedding_score": score}


def _embedding_sorted(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        docs,
        key=lambda item: float(item.get("embedding_score", item.get("score", 0.0))),
        reverse=True,
    )


def _top_k(top_k: Optional[int]) -> int:
    if top_k is None:
        top_k = settings.annotation_label_reranker_top_k
    return max(0, int(top_k))


def _label_doc_text(doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("id", "name", "type", "parent", "description", "annotation_guideline"):
        value = doc.get(key)
        if value is not None and str(value).strip():
            parts.append(f"{key}: {value}")
    for key, value in doc.items():
        if key.startswith("_") or key in {
            "id",
            "name",
            "type",
            "parent",
            "description",
            "annotation_guideline",
            "score",
            "embedding_score",
            "reranker_score",
        }:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


__all__ = ["rerank_label_docs"]
