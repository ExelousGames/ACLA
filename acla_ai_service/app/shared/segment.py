"""Segment dataclasses + the feature aggregator catalog.

Pure domain types: ``AnnotatedSegment`` (a labelled telemetry slice
produced by the annotation pipeline) and ``PredictedSegment`` (a nested
behavior or sub-label detection emitted by the temporal classifier).

``SegmentFeatureCatalog`` aggregates feature-name lists from the
three upstream catalogs (telemetry, expert, tire-grip). All three
sources live in app/shared/ so this file is a true leaf.

Moved from app/models/segment_models.py in refactor/hexagonal-v1, Step 3.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.shared.expert_features import ExpertFeatureCatalog
from app.shared.telemetry import TelemetryFeatures
from app.shared.tire_grip_features import TireGripFeatureCatalog


@dataclass
class AnnotatedSegment:
    labels: List[str]
    segment_length: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    chunk_index: Optional[int] = None
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None
    parent_id: Optional[str] = None
    opponent_interaction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotatedSegment':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            labels=data.get("labels", []),
            segment_length=data.get("segment_length", 0),
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
            chunk_index=data.get("chunk_index"),
            telemetry_data=data.get("telemetry_data", []),
            notes=data.get("notes"),
            parent_id=data.get("parent_id"),
            opponent_interaction=data.get("opponent_interaction"),
        )


@dataclass
class PredictedSegment:
    label: str
    score: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    subsegments: List[PredictedSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SegmentFeatureCatalog:
    """
    Centralized catalog for all features used in segment analysis and modeling.
    Aggregates features from various services and models.
    """

    @staticmethod
    def get_all_available_features() -> List[str]:
        """
        Returns a unique list of all features available for segment analysis.
        Combines base telemetry, expert optimal, expert context, and tire grip context.
        """
        features = []
        features.extend(TelemetryFeatures.get_features_for_top_lap_reference())
        features.extend([f.value for f in ExpertFeatureCatalog.ExpertFeatures])
        features.extend([f.value for f in TireGripFeatureCatalog.ContextFeature])

        # Use dict.fromkeys to remove duplicates while preserving insertion order
        return list(dict.fromkeys(features))


__all__ = [
    "AnnotatedSegment",
    "PredictedSegment",
    "SegmentFeatureCatalog",
]
