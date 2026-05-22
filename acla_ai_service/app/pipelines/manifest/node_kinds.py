"""Registry of annotation and training node kinds for the Pipeline UI.

Adding a new kind = append one entry to the ``register(...)`` calls below.
The Pipeline graph view and the "kind" dropdown on each node pick it up
automatically; no other file needs to change.

Conventions
-----------
``ui_route`` — equal to ``kind`` by convention. The shell dispatches on
    ``node.kind`` to pick which tab renderer runs, and writes the node's
    resolved keys into ``st.session_state`` so the tab loads them.

``produces_output`` — annotation kinds set this True; the pipeline auto-
    grows its output-dataset list when the node's ``output_key`` first
    appears in the Lance store. Training kinds set False (they emit a
    model directory, not a dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal


Category = Literal["annotation", "training"]


@dataclass(frozen=True)
class NodeKindSpec:
    kind: str
    category: Category
    display: str
    description: str
    ui_route: str
    produces_output: bool = False


_REGISTRY: Dict[str, NodeKindSpec] = {}

# Legacy kind aliases — manifests on disk may still use the old strings.
_ALIASES: Dict[str, str] = {
    "parent": "lap",
    "children": "detailed",
    "batch": "batch_subseg",
}


def register(spec: NodeKindSpec) -> None:
    if spec.kind in _REGISTRY:
        raise ValueError(f"NodeKindSpec already registered: {spec.kind}")
    _REGISTRY[spec.kind] = spec


def canonicalize(kind: str) -> str:
    """Map a legacy kind string to its current name (or return unchanged)."""
    return _ALIASES.get(kind, kind)


def get(kind: str) -> NodeKindSpec:
    canonical = canonicalize(kind)
    if canonical not in _REGISTRY:
        raise KeyError(f"Unknown node kind: {kind}")
    return _REGISTRY[canonical]


def list_by_category(category: Category) -> List[NodeKindSpec]:
    return [s for s in _REGISTRY.values() if s.category == category]


# ── Annotation kinds ──────────────────────────────────────────────────────
register(NodeKindSpec(
    kind="lap",
    category="annotation",
    display="Lap Annotation",
    description="Manual lap segmentation + main-label tagging.",
    ui_route="lap",
    produces_output=True,
))
register(NodeKindSpec(
    kind="detailed",
    category="annotation",
    display="Detailed Annotation",
    description="Sub-segment / sub-label refinement on top of lap segments.",
    ui_route="detailed",
    produces_output=True,
))
register(NodeKindSpec(
    kind="batch_bulk_label",
    category="annotation",
    display="Batch — Bulk Label Mgmt",
    description="Remove a label from every segment in one click.",
    ui_route="batch_bulk_label",
    produces_output=True,
))
register(NodeKindSpec(
    kind="batch_rule_based",
    category="annotation",
    display="Batch — Rule-Based",
    description="Apply a label to segments where a feature matches a value.",
    ui_route="batch_rule_based",
    produces_output=True,
))
register(NodeKindSpec(
    kind="batch_classifier",
    category="annotation",
    display="Batch — Classifier Auto",
    description="Identify segments using the trained LSTM classifier.",
    ui_route="batch_classifier",
    produces_output=True,
))
register(NodeKindSpec(
    kind="batch_subseg",
    category="annotation",
    display="Batch — Sub-Segment Discovery",
    description="Bulk discover children via Local VLM or Claude.",
    ui_route="batch_subseg",
    produces_output=True,
))
register(NodeKindSpec(
    kind="batch_lap",
    category="annotation",
    display="Batch — Lap-to-Segment Excerpter",
    description="Bulk Claude lap → per-circuit-section annotation.",
    ui_route="batch_lap",
    produces_output=True,
))
register(NodeKindSpec(
    kind="llm",
    category="annotation",
    display="LLM Annotation",
    description="Claude critique/guide draft generation for training units.",
    ui_route="llm",
    produces_output=True,
))


# ── Training kinds ────────────────────────────────────────────────────────
register(NodeKindSpec(
    kind="classifier",
    category="training",
    display="Classifier Training",
    description="LSTM segment classifier.",
    ui_route="classifier",
))
register(NodeKindSpec(
    kind="transformer",
    category="training",
    display="Transformer Training",
    description="Transformer guidance head.",
    ui_route="transformer",
))
register(NodeKindSpec(
    kind="llm_training",
    category="training",
    display="LLM Training",
    description="LLM fine-tune on chat-format JSONL.",
    ui_route="llm_training",
))


__all__ = ["NodeKindSpec", "register", "get", "list_by_category", "canonicalize"]
