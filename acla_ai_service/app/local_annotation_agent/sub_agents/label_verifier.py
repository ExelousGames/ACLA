"""Knowledge-backed label shortlist helper.

The local harness no longer registers graph sub-agents. This module keeps the
useful part of the old label verifier: a pure retrieval/reranking helper that
can be called by flows or future dedicated harness workers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.infra.config import settings
from app.internal_knowledge_base.label_reranker import rerank_label_docs
from app.internal_knowledge_base.label_search import get_doc, search
from app.shared.contracts import Attachment

LOGGER = logging.getLogger(__name__)

LABEL_VERIFIER_AGENT_NAME = "label_verifier"

_MAX_VERIFIED = 8


def _payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    score = float(doc.get("score", 0.0))
    payload = {
        "label_id": doc["id"],
        "name": doc.get("name", doc["id"]),
        "description": doc.get("description", ""),
        "similarity": score,
    }
    if "embedding_score" in doc:
        payload["embedding_similarity"] = float(doc.get("embedding_score", 0.0))
    if "reranker_score" in doc:
        payload["reranker_score"] = float(doc.get("reranker_score", 0.0))
    return payload


def compute_verified_labels(
    parent_main_labels: List[str],
    evidence_text: str,
    eligible_behavior_label_ids: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shortlist eligible labels by hybrid similarity to the evidence prose.

    Returns ``(verified, all_scored)``. With reranking enabled, both contain the
    final post-rerank shortlist; with reranking disabled, this preserves the
    embedding-only top-N behavior.

    The eligible tiers depend on the flow, read off the given parents:

    - a real ``type == "main"`` parent (detailed flow) ⇒ records whose
      ``parent`` field is that label id, plus segment types,
    - no main parent (lap flow — the main label is still being
      discovered) ⇒ the main labels + segment types.
    """
    query = (evidence_text or "").strip()
    if not query:
        return [], []

    main_parents = [
        p for p in parent_main_labels if (get_doc(p) or {}).get("type") == "main"
    ]

    merged: Dict[str, Tuple[Dict[str, Any], float]] = {}

    eligible_behavior_label_ids = list(eligible_behavior_label_ids or [])

    def _allowed(doc: Dict[str, Any]) -> bool:
        if not eligible_behavior_label_ids:
            return True
        required_parents = {"O", "OD", "PS", "RM", "MSP", "MSR"}
        lid = str(doc.get("id") or "")
        parent = str(doc.get("parent") or "")
        if lid in required_parents:
            return lid in eligible_behavior_label_ids
        if parent in required_parents:
            return parent in eligible_behavior_label_ids
        return True

    def _absorb(docs: List[Dict[str, Any]]) -> None:
        for d in docs:
            if not _allowed(d):
                continue
            lid = d["id"]
            score = float(d.get("score", 0.0))
            if lid not in merged or score > merged[lid][1]:
                merged[lid] = (d, score)

    _absorb(search(query, filters={"type": "segment_type"}, top_k=_MAX_VERIFIED))
    if main_parents:
        for pid in main_parents:
            _absorb(search(query, filters={"parent": pid}, top_k=_MAX_VERIFIED))
    else:
        _absorb(search(query, filters={"type": "main"}, top_k=_MAX_VERIFIED))

    docs = [doc for doc, _score in merged.values()]
    top_k = (
        settings.annotation_label_reranker_top_k
        if settings.annotation_label_reranker_enabled
        else _MAX_VERIFIED
    )
    scored = rerank_label_docs(query, docs, top_k=top_k)
    all_scored = [_payload(d) for d in scored]
    verified = [_payload(d) for d in scored]
    return verified, all_scored


AttachmentPool = Dict[str, Attachment]


def evidence_text_from_pool(pool: AttachmentPool) -> str:
    """Concatenate preflight evidence and all ``*.observations`` attachments."""
    parts: List[str] = []
    preflight = pool.get("init.annotation_preflight_context")
    if preflight and isinstance(preflight.content, dict):
        semantic_text = (
            preflight.content.get("semantic_evidence_text")
            or preflight.content.get("semantic_search_text")
        )
        if semantic_text:
            parts.append(str(semantic_text))
    for name in sorted(pool.keys()):
        if not name.endswith(".observations"):
            continue
        att = pool[name]
        c = att.content if isinstance(att.content, dict) else {}
        obs = c.get("graph_observations")
        if obs:
            parts.append(str(obs))
    return " ".join(parts).strip()


def parent_main_labels_from_pool(pool: AttachmentPool) -> List[str]:
    """Read the parent's candidate main labels out of init.parent_segment.

    The annotation flow seeds this attachment in ``build_request``; the
    content is ``{"main_labels": [...], ...}``.
    """
    att = pool.get("init.parent_segment")
    if not att or not isinstance(att.content, dict):
        return []
    raw = att.content.get("main_labels") or []
    return [str(x) for x in raw if isinstance(x, str)]


def eligible_behavior_label_ids_from_pool(pool: AttachmentPool) -> List[str]:
    att = pool.get("init.parent_segment")
    if not att or not isinstance(att.content, dict):
        return []
    raw = att.content.get("eligible_behavior_label_ids") or []
    return [str(x) for x in raw if isinstance(x, str)]


def _emit_verified(payload: List[dict]) -> Attachment:
    return Attachment(
        name="label_verifier.verified_labels",
        kind="structured",
        content_schema="verified_labels",
        label="Verified Labels",
        content=payload,
    )


def run_label_verifier(pool: AttachmentPool) -> Dict[str, Any]:
    """Run label retrieval against an attachment pool."""
    parent_main_labels = parent_main_labels_from_pool(pool)
    eligible_behavior_label_ids = eligible_behavior_label_ids_from_pool(pool)
    evidence = evidence_text_from_pool(pool)

    verified, all_scored = compute_verified_labels(
        parent_main_labels,
        evidence,
        eligible_behavior_label_ids=eligible_behavior_label_ids,
    )

    if not evidence:
        LOGGER.warning("Label retrieval: no evidence text; empty shortlist.")
        message = "No evidence text available; emitted an empty shortlist."
    else:
        passed_log = "\n".join(
            f"✓ {p['label_id']} ({p['name']}): {p['similarity']:.3f}"
            for p in verified
        )
        verified_ids = {p["label_id"] for p in verified}
        rejected_log = "\n".join(
            f"✗ {p['label_id']} ({p['name']}): {p['similarity']:.3f}"
            for p in all_scored
            if p["label_id"] not in verified_ids
        )
        LOGGER.info(
            "Label retrieval: %d/%d candidates shortlisted.",
            len(verified), len(all_scored),
        )
        message = (
            f"Hybrid retrieval: {len(verified)}/{len(all_scored)} labels shortlisted:\n"
            f"{passed_log}"
            + (f"\n\nNot shortlisted:\n{rejected_log}" if rejected_log else "")
        )

    att = _emit_verified(verified)
    return {
        "attachment_pool": {att.name: att},
        "messages": [{"role": "label_verifier", "content": message}],
    }
