"""label_verifier sub-agent — embedding-similarity label filter.

Sits alongside ``describe_graphs`` and ``zoom`` as a peer capability the
agent box exposes. Two surfaces share the same core computation:

  * ``compute_verified_labels(parent_main_labels, evidence_text)`` —
    pure function the local runner's synth phase calls when the VLM
    invokes the ``verify_labels`` tool.
  * ``LabelVerifier`` Agent — registered with the framework so the
    LangGraph planner can still delegate a step to it when a flow
    prefers the deterministic-plan path over a VLM tool call.

Consumes  init.parent_segment, step_solver.*.observations
Produces  verified_labels
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.domain.labels import LABEL_MAPPING
from app.skills.annotation import embed
from app.skills.annotation.label_lookup import find_labels, get_label
from app.agents.framework import Agent, AgentState
from app.agents.evaluators import (
    AttachmentPool,
    PipelineAttachment,
)

LOGGER = logging.getLogger(__name__)

LABEL_VERIFIER_AGENT_NAME = "label_verifier"


_SIMILARITY_THRESHOLD = 0.25
_MIN_FILTER_LABELS = 2
_MAX_FILTER_LABELS = 8


def _shortlist_candidate_ids(parent_main_labels: List[str]) -> List[str]:
    """Return the deduplicated candidate label IDs the filter scores."""
    candidate_ids: List[str] = []
    for pid in parent_main_labels:
        for entry in find_labels(parent=pid):
            candidate_ids.append(entry["id"])
    for entry in find_labels(type="segment_type"):
        candidate_ids.append(entry["id"])

    seen: set = set()
    shortlisted: List[str] = []
    for lid in candidate_ids:
        if lid not in seen:
            seen.add(lid)
            shortlisted.append(lid)
    return shortlisted


def _entry_to_payload(lid: str, similarity: Any) -> Dict[str, Any]:
    entry = get_label(lid)
    return {
        "label_id": lid,
        "name": entry["name"] if entry else LABEL_MAPPING.get(lid, lid),
        "description": entry.get("description", "") if entry else "",
        "similarity": similarity,
    }


def compute_verified_labels(
    parent_main_labels: List[str],
    evidence_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score the parent's candidate labels against the evidence prose.

    Returns ``(verified, all_scored)`` — verified is the filtered
    shortlist (above threshold or top-N), all_scored carries every
    candidate's similarity for diagnostics.
    """
    import numpy as np

    shortlisted = _shortlist_candidate_ids(parent_main_labels)
    if not shortlisted:
        return [], []

    query_text = (evidence_text or "").strip()
    if not query_text:
        fallback = shortlisted[:_MAX_FILTER_LABELS]
        return [_entry_to_payload(lid, None) for lid in fallback], []

    label_texts: List[str] = []
    for lid in shortlisted:
        entry = get_label(lid)
        if entry:
            desc = entry.get("description", "")
            text = f"{entry['name']}: {desc}" if desc else entry["name"]
            label_texts.append(text)
        else:
            label_texts.append(LABEL_MAPPING.get(lid, lid))

    query_emb: np.ndarray = embed(query_text)
    label_embs: np.ndarray = embed(label_texts)

    scored: List[Tuple[str, float]] = [
        (lid, float(query_emb @ label_embs[i]))
        for i, lid in enumerate(shortlisted)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    above = [(lid, sim) for lid, sim in scored if sim >= _SIMILARITY_THRESHOLD]
    if len(above) < _MIN_FILTER_LABELS:
        above = scored[:_MIN_FILTER_LABELS]
    filtered = above[:_MAX_FILTER_LABELS]

    verified = [_entry_to_payload(lid, sim) for lid, sim in filtered]
    all_scored = [_entry_to_payload(lid, sim) for lid, sim in scored]
    return verified, all_scored


def evidence_text_from_pool(pool: AttachmentPool) -> str:
    """Concatenate ``graph_observations`` from every ``*.observations`` attachment."""
    parts: List[str] = []
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


def _emit_verified(payload: List[dict]) -> PipelineAttachment:
    return PipelineAttachment(
        name="label_verifier.verified_labels",
        kind="structured",
        content_schema="verified_labels",
        label="Verified Labels",
        content=payload,
    )


def _executor(state: AgentState, step: Dict[str, Any], registry) -> Dict[str, Any]:
    """Agent executor surface — wraps ``compute_verified_labels`` against the pool."""
    messages = list(state.get("messages", []))
    pool: AttachmentPool = state.get("attachment_pool", {})
    parent_main_labels = parent_main_labels_from_pool(pool)
    evidence = evidence_text_from_pool(pool)

    verified, all_scored = compute_verified_labels(parent_main_labels, evidence)

    if not _shortlist_candidate_ids(parent_main_labels):
        LOGGER.info("Label similarity filter: no candidate labels.")
        messages.append({
            "role": "label_verifier",
            "content": "No candidate labels available for parent categories.",
        })
        att = _emit_verified([])
        return {
            "attachment_pool": {att.name: att},
            "messages": messages,
        }

    if not evidence:
        LOGGER.warning("Label similarity filter: no evidence text; passing top candidates.")
        messages.append({
            "role": "label_verifier",
            "content": (
                f"No evidence text; passed top {len(verified)} candidates unchanged."
            ),
        })
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
            "Label similarity filter: %d/%d passed (threshold=%.2f)",
            len(verified), len(all_scored), _SIMILARITY_THRESHOLD,
        )
        messages.append({
            "role": "label_verifier",
            "content": (
                f"Embedding filter: {len(verified)}/{len(all_scored)} labels passed "
                f"(threshold={_SIMILARITY_THRESHOLD}):\n{passed_log}"
                + (f"\n\nRejected:\n{rejected_log}" if rejected_log else "")
            ),
        })

    att = _emit_verified(verified)
    return {
        "attachment_pool": {att.name: att},
        "messages": messages,
    }


class LabelVerifier(Agent):
    """Deterministic step solver: embedding-similarity filter."""

    name = LABEL_VERIFIER_AGENT_NAME
    consumes = ["init.parent_segment", "step_solver.*.observations"]
    produces = ["verified_labels"]
    delegates_to: list = []

    def planner(self, state: AgentState):
        return None

    def synthesizer(self, state: AgentState):
        return None

    def evaluator(self, state: AgentState):
        return None

    def executor(self, state: AgentState, step, registry) -> Dict[str, Any]:
        return _executor(state, step, registry)


LABEL_VERIFIER_SPEC = LabelVerifier.register()
