"""
label_verifier leaf Agent — embedding-similarity label filter.

Deterministic step solver (no planner, no synthesizer, no evaluator). Reads
every describe_graphs observation attachment in its sliced pool, embeds the
concatenated text, scores the candidate labels (parent's sub-labels + segment
types), and emits a verified_labels attachment for the synthesizer to consume.

Consumes  init.parent_segment, step_solver.*.observations
Produces  verified_labels
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from app.models.label_catalog import get_label_catalog
from app.models.segment_models import LABEL_MAPPING
from app.services.llm.agent_framework import Agent, AgentState
from app.services.llm.step_evaluator_agents import (
    AttachmentPool,
    PipelineAttachment,
)

LOGGER = logging.getLogger(__name__)

LABEL_VERIFIER_AGENT_NAME = "label_verifier"


# ---------------------------------------------------------------------------
# Embedding model singleton (lazy-loaded, shared across runs)
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.25
_MIN_FILTER_LABELS = 2   # always pass at least this many (highest scoring)
_MAX_FILTER_LABELS = 8   # cap to keep synthesizer prompt focused

_embedder_instance = None
_embedder_lock = threading.Lock()


def _get_embedder():
    """Return the SentenceTransformer singleton, loading on first call."""
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance
    with _embedder_lock:
        if _embedder_instance is None:
            from sentence_transformers import SentenceTransformer
            LOGGER.info("Loading embedding model '%s' …", _EMBED_MODEL_NAME)
            _embedder_instance = SentenceTransformer(_EMBED_MODEL_NAME)
            LOGGER.info("Embedding model loaded.")
    return _embedder_instance


def _cosine_sim(a, b) -> float:
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _emit_verified(payload: List[dict]) -> PipelineAttachment:
    return PipelineAttachment(
        name="label_verifier.verified_labels",
        kind="structured",
        content_schema="verified_labels",
        label="Verified Labels",
        content=payload,
    )


class LabelVerifier(Agent):
    """Deterministic step solver: embedding-similarity filter.

    Has no planner / synthesizer / evaluator — all three explicitly return
    None to mark the phases as skipped, satisfying the contract.
    """

    name = LABEL_VERIFIER_AGENT_NAME
    consumes = ["init.parent_segment", "step_solver.*.observations"]
    produces = ["verified_labels"]
    delegates_to: list = []

    def planner(self, state: AgentState):
        return None     # deterministic — no planning needed

    def synthesizer(self, state: AgentState):
        return None     # single-step — nothing to merge

    def evaluator(self, state: AgentState):
        return None     # deterministic — no verdict needed

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return _executor(state, step, registry)


def _executor(state: AgentState, step: Dict[str, Any], registry) -> Dict[str, Any]:
    """Embedding-similarity filter over candidate labels.

    Reads observation attachments from the pool (sliced to
    ``step_solver.*.observations`` by the framework), concatenates their
    prose, scores each candidate label by cosine similarity, and emits the
    survivors as ``label_verifier.verified_labels``.
    """
    import numpy as np

    messages = list(state.get("messages", []))
    parent_main_labels = state.get("parent_main_labels", [])

    catalog = get_label_catalog()
    candidate_ids: List[str] = []
    for pid in parent_main_labels:
        for entry in catalog.get_sublabels(pid):
            candidate_ids.append(entry.id)
    for entry in catalog.get_segment_types():
        candidate_ids.append(entry.id)

    seen: set = set()
    shortlisted: List[str] = []
    for lid in candidate_ids:
        if lid not in seen:
            seen.add(lid)
            shortlisted.append(lid)

    if not shortlisted:
        LOGGER.info("Label similarity filter: no candidate labels.")
        messages.append({
            "role": "label_verifier",
            "content": "No candidate labels available for parent categories.",
        })
        att = _emit_verified([])
        return {
            "verified_labels": [],
            "verified_label_reasoning": {},
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    # Sliced pool gives us all step_solver.*.observations attachments.
    pool: AttachmentPool = state.get("attachment_pool", {})
    evidence_parts: List[str] = []
    for name in sorted(pool.keys()):
        if not name.endswith(".observations"):
            continue
        att = pool[name]
        c = att.content if isinstance(att.content, dict) else {}
        obs = c.get("graph_observations")
        if obs:
            evidence_parts.append(str(obs))
    query_text = " ".join(evidence_parts).strip()

    if not query_text:
        LOGGER.warning("Label similarity filter: no evidence text; passing top candidates.")
        fallback = shortlisted[:_MAX_FILTER_LABELS]
        messages.append({
            "role": "label_verifier",
            "content": f"No evidence text; passed top {len(fallback)} candidates unchanged.",
        })
        fallback_payload = []
        for lid in fallback:
            entry = catalog.get_label(lid)
            fallback_payload.append({
                "label_id": lid,
                "name": entry.name if entry else LABEL_MAPPING.get(lid, lid),
                "description": entry.description if entry else "",
                "similarity": None,
            })
        att = _emit_verified(fallback_payload)
        return {
            "verified_labels": fallback,
            "verified_label_reasoning": {lid: "No evidence available." for lid in fallback},
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    label_texts: List[str] = []
    for lid in shortlisted:
        entry = catalog.get_label(lid)
        if entry:
            text = f"{entry.name}: {entry.description}" if entry.description else entry.name
            label_texts.append(text)
        else:
            label_texts.append(LABEL_MAPPING.get(lid, lid))

    embedder = _get_embedder()
    query_emb: np.ndarray = embedder.encode(query_text, convert_to_numpy=True)
    label_embs: np.ndarray = embedder.encode(label_texts, convert_to_numpy=True)

    scored: List[tuple] = [
        (lid, _cosine_sim(query_emb, label_embs[i]), label_texts[i])
        for i, lid in enumerate(shortlisted)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    above = [(lid, sim, txt) for lid, sim, txt in scored if sim >= _SIMILARITY_THRESHOLD]
    if len(above) < _MIN_FILTER_LABELS:
        above = scored[:_MIN_FILTER_LABELS]
    filtered = above[:_MAX_FILTER_LABELS]

    verified: List[str] = [lid for lid, _, _ in filtered]
    reasoning_map: Dict[str, str] = {
        lid: f"Similarity {sim:.3f} — {txt}"
        for lid, sim, txt in filtered
    }

    passed_log = "\n".join(
        f"✓ {lid} ({LABEL_MAPPING.get(lid, lid)}): {sim:.3f}"
        for lid, sim, _ in filtered
    )
    rejected_log = "\n".join(
        f"✗ {lid} ({LABEL_MAPPING.get(lid, lid)}): {sim:.3f}"
        for lid, sim, _ in scored
        if lid not in verified
    )

    LOGGER.info(
        "Label similarity filter: %d/%d passed (threshold=%.2f)",
        len(verified), len(shortlisted), _SIMILARITY_THRESHOLD,
    )
    messages.append({
        "role": "label_verifier",
        "content": (
            f"Embedding filter: {len(verified)}/{len(shortlisted)} labels passed "
            f"(threshold={_SIMILARITY_THRESHOLD}):\n{passed_log}"
            + (f"\n\nRejected:\n{rejected_log}" if rejected_log else "")
        ),
    })

    verified_payload: List[dict] = []
    for lid, sim, _txt in filtered:
        entry = catalog.get_label(lid)
        verified_payload.append({
            "label_id": lid,
            "name": entry.name if entry else LABEL_MAPPING.get(lid, lid),
            "description": entry.description if entry else "",
            "similarity": sim,
        })

    att = _emit_verified(verified_payload)
    return {
        "verified_labels": verified,
        "verified_label_reasoning": reasoning_map,
        "attachment_pool": {att.name: att},
        "messages": messages,
    }


LABEL_VERIFIER_SPEC = LabelVerifier.register()
