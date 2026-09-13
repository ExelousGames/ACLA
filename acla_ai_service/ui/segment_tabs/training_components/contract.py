"""Opening contract for Streamlit training components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.pipelines.manifest.models import TrainingNode


@dataclass(frozen=True)
class TrainingComponentContext:
    node: TrainingNode | None
    input_key: str | None


class TrainingComponent(ABC):
    @abstractmethod
    def open(self, context: TrainingComponentContext) -> None:
        """Render the component for the selected training node."""


__all__ = ["TrainingComponent", "TrainingComponentContext"]
