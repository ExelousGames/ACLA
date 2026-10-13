"""Training component contract and registry."""

from .contract import TrainingComponent, TrainingComponentContext
from .registry import (
    TRAINING_COMPONENTS,
    TRAINING_ROUTES,
    build_training_component_registry,
    get_training_component,
)

__all__ = [
    "TrainingComponent",
    "TrainingComponentContext",
    "TRAINING_COMPONENTS",
    "TRAINING_ROUTES",
    "build_training_component_registry",
    "get_training_component",
]
