"""label_verifier sub-agent — hybrid-retrieval label shortlist.

Sits alongside ``describe_graphs`` and ``zoom`` as a peer capability the
agent box exposes. Two surfaces share the same core computation:

  * ``compute_verified_labels(parent_main_labels, evidence_text)`` —
    pure function the local runner's synth phase calls.
  * ``LabelVerifier`` Agent — registered with the framework so the
    LangGraph planner can delegate a plan step to it.

The shortlist comes from the one hybrid retriever in
:mod:`app.skills.internal.annotation.label_search`; this agent just queries it
with the describe_graphs observations and scopes the result to the
eligible tiers.

Consumes  init.parent_segment, step_solver.*.observations
Produces  verified_labels
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.skills.internal.annotation.label_search import get_doc, search
from app.agents.framework import Agent, AgentState
from app.agents.evaluators import (
    AttachmentPool,
    PipelineAttachment,
)

LOGGER = logging.getLogger(__name__)

LABEL_VERIFIER_AGENT_NAME = "label_verifier"

_MAX_VERIFIED = 8


def _payload(doc: Dict[str, Any], score: float) -> Dict[str, Any]:
    return {
        "label_id": doc["id"],
        "name": doc.get("name", doc["id"]),
        "description": doc.get("description", ""),
        "similarity": score,
    }


def compute_verified_labels(
    parent_main_labels: List[str],
    evidence_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shortlist eligible labels by hybrid similarity to the evidence prose.

    Returns ``(verified, all_scored)`` — verified is the top-N shortlist,
    all_scored carries every retrieved candidate for diagnostics.

    The eligible tiers depend on the flow, read off the given parents:

    - a real ``type == "main"`` parent (detailed flow) ⇒ that parent's
      sub-labels + segment types,
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

    def _absorb(docs: List[Dict[str, Any]]) -> None:
        for d in docs:
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

    scored = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    all_scored = [_payload(d, s) for d, s in scored]
    verified = [_payload(d, s) for d, s in scored[:_MAX_VERIFIED]]
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

    if not evidence:
        LOGGER.warning("Label retrieval: no evidence text; empty shortlist.")
        messages.append({
            "role": "label_verifier",
            "content": "No evidence text available; emitted an empty shortlist.",
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
            "Label retrieval: %d/%d candidates shortlisted.",
            len(verified), len(all_scored),
        )
        messages.append({
            "role": "label_verifier",
            "content": (
                f"Hybrid retrieval: {len(verified)}/{len(all_scored)} labels shortlisted:\n"
                f"{passed_log}"
                + (f"\n\nNot shortlisted:\n{rejected_log}" if rejected_log else "")
            ),
        })

    att = _emit_verified(verified)
    return {
        "attachment_pool": {att.name: att},
        "messages": messages,
    }


class LabelVerifier(Agent):
    """Deterministic step solver: hybrid-retrieval label shortlist."""

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
