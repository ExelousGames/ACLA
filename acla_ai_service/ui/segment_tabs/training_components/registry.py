"""Validated registry of Streamlit training components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from app.pipelines.manifest import node_kinds
from app.pipelines.manifest.node_kinds import NodeKindSpec

from .classifier import ClassifierTrainingComponent
from .contract import TrainingComponent
from .opportunity_forecaster import OpportunityForecasterTrainingComponent
from .segment_cropper import SegmentCropperTrainingComponent
from .transformer import TransformerTrainingComponent


def build_training_component_registry(
    registrations: Iterable[tuple[str, TrainingComponent]],
    *,
    training_specs: Sequence[NodeKindSpec] | None = None,
) -> Mapping[str, TrainingComponent]:
    specs = list(
        training_specs
        if training_specs is not None
        else node_kinds.list_by_category("training")
    )
    specs_by_kind = {spec.kind: spec for spec in specs}
    if len(specs_by_kind) != len(specs):
        raise ValueError("Duplicate training kind in node_kinds")

    components_by_kind: dict[str, TrainingComponent] = {}
    component_ids: set[int] = set()
    for kind, component in registrations:
        if kind in components_by_kind:
            raise ValueError(f"Duplicate training component registration: {kind}")
        if not isinstance(component, TrainingComponent):
            raise TypeError(
                f"Training component for {kind} does not implement the contract"
            )
        if id(component) in component_ids:
            raise ValueError("Duplicate training component instance registration")
        components_by_kind[kind] = component
        component_ids.add(id(component))

    expected_kinds = set(specs_by_kind)
    registered_kinds = set(components_by_kind)
    missing = sorted(expected_kinds - registered_kinds)
    unexpected = sorted(registered_kinds - expected_kinds)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            "Training component registrations do not match node_kinds ("
            + "; ".join(details)
            + ")"
        )

    components_by_route: dict[str, TrainingComponent] = {}
    for spec in specs:
        if spec.ui_route in components_by_route:
            raise ValueError(f"Duplicate training route registration: {spec.ui_route}")
        components_by_route[spec.ui_route] = components_by_kind[spec.kind]

    return MappingProxyType(components_by_route)


_REGISTRATIONS = (
    ("classifier", ClassifierTrainingComponent()),
    ("segment_cropper", SegmentCropperTrainingComponent()),
    ("transformer", TransformerTrainingComponent()),
    ("opportunity_forecaster", OpportunityForecasterTrainingComponent()),
)

TRAINING_COMPONENTS = build_training_component_registry(_REGISTRATIONS)
TRAINING_ROUTES = frozenset(TRAINING_COMPONENTS)


def get_training_component(route: str) -> TrainingComponent:
    try:
        return TRAINING_COMPONENTS[route]
    except KeyError:
        raise KeyError(f"Unknown training route: {route}") from None


__all__ = [
    "TRAINING_COMPONENTS",
    "TRAINING_ROUTES",
    "build_training_component_registry",
    "get_training_component",
]
