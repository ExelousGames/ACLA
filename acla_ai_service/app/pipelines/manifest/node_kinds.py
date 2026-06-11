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

from dataclasses import dataclass, field
from typing import Dict, List, Literal


Category = Literal["annotation", "training"]
InputDatasetShape = Literal["records", "segments"]


@dataclass(frozen=True)
class DatasetFieldSpec:
    field: str
    type: str


@dataclass(frozen=True)
class NodeKindSpec:
    kind: str
    category: Category
    display: str
    description: str
    ui_route: str
    produces_output: bool = False
    input_dataset_shape: InputDatasetShape = "records"
    output_fields: tuple[DatasetFieldSpec, ...] = field(default_factory=tuple)


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


ANNOTATED_SEGMENT_OUTPUT_FIELDS = (
    DatasetFieldSpec("id", "str"),
    DatasetFieldSpec("labels", "list[str]"),
    DatasetFieldSpec("segment_length", "int"),
    DatasetFieldSpec("start_index", "int | None"),
    DatasetFieldSpec("end_index", "int | None"),
    DatasetFieldSpec("chunk_index", "int | str | None"),
    DatasetFieldSpec("telemetry_data", "list[input row]"),
    DatasetFieldSpec("notes", "str | None"),
    DatasetFieldSpec("parent_id", "str | None"),
    DatasetFieldSpec("opponent_interaction", "dict | None"),
)


def annotation_spec(
    *,
    kind: str,
    display: str,
    description: str,
    ui_route: str,
    input_dataset_shape: InputDatasetShape = "records",
) -> NodeKindSpec:
    return NodeKindSpec(
        kind=kind,
        category="annotation",
        display=display,
        description=description,
        ui_route=ui_route,
        produces_output=True,
        input_dataset_shape=input_dataset_shape,
        output_fields=ANNOTATED_SEGMENT_OUTPUT_FIELDS,
    )


# ── Annotation kinds ──────────────────────────────────────────────────────
register(annotation_spec(
    kind="lap",
    display="Lap Annotation",
    description="Manual lap segmentation + main-label tagging.",
    ui_route="lap",
))
register(annotation_spec(
    kind="detailed",
    display="Detailed Annotation",
    description="Sub-segment / sub-label refinement on top of lap segments.",
    ui_route="detailed",
    input_dataset_shape="segments",
))
register(annotation_spec(
    kind="batch_bulk_label",
    display="Batch — Bulk Label Mgmt",
    description="Remove a label from every segment in one click.",
    ui_route="batch_bulk_label",
))
register(annotation_spec(
    kind="batch_rule_based",
    display="Batch — Rule-Based",
    description="Apply a label to segments where a feature matches a value.",
    ui_route="batch_rule_based",
))
register(annotation_spec(
    kind="batch_classifier",
    display="Batch — Classifier Auto",
    description="Identify segments using the trained LSTM classifier.",
    ui_route="batch_classifier",
))
register(annotation_spec(
    kind="batch_subseg",
    display="Batch — Sub-Segment Discovery",
    description="Bulk discover children via Local VLM or Claude.",
    ui_route="batch_subseg",
    input_dataset_shape="segments",
))
register(annotation_spec(
    kind="batch_lap",
    display="Batch — Lap-to-Segment Excerpter",
    description="Bulk Claude lap → per-circuit-section annotation.",
    ui_route="batch_lap",
))
register(annotation_spec(
    kind="parent_labels",
    display="Parent Label Propagation",
    description="Append missing parent labels to child sub-segments.",
    ui_route="parent_labels",
    input_dataset_shape="segments",
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
    kind="opportunity_forecaster",
    category="training",
    display="Opportunity Forecaster Training",
    description="Future successful overtake / defense probability model.",
    ui_route="opportunity_forecaster",
))
register(NodeKindSpec(
    kind="llm_training",
    category="training",
    display="LLM Training",
    description="LLM fine-tune on chat-format JSONL.",
    ui_route="llm_training",
))


__all__ = [
    "DatasetFieldSpec",
    "InputDatasetShape",
    "NodeKindSpec",
    "register",
    "get",
    "list_by_category",
    "canonicalize",
]
