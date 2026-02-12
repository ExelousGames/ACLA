from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from app.models.telemetry_models import TelemetryFeatures
from app.services.imitate_expert_learning_service import ExpertFeatureCatalog
from app.services.tire_grip_analysis_service import TireGripFeatureCatalog

# Constants
LABEL_MAPPING = {
    ################### Main Labels ###################
    "1": "Overtaking",
    "2": "Missing data",
    "3": "Expert Adherence",
    "4": "Pit Stop",
    "5": "Recovery & Merge",
    "MS" :"Mistake segment",
    ################### Detailed Expert Adherence Labels (for label 3) ###################
    "3001": "",
    ################### Detailed mistake labels (for label MS) ###################
    "MS1": "Brake too late",
    "MS2": "Turn in too late",
    "MS3": "Apex too late",
    "MS4": "Exit out too early",
    "MS5": "Brake too early",
    "MS6": "Turn in too early",
    "MS7": "Apex too early",
    "MS8": "Exit out too late",
    "MS9": "Entry not wide enough",
    "MS10": "Apex not tight enough",
    "MS11": "Exit not wide enough",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

LABEL_CATEGORIES = {
    "Main Labels": ["1", "2","3","4","5","MS"],
    "1":[],
    "2":[],
    "3":[],
    "4":[],
    "5":[],
    "MS":["MS1","MS2","MS3","MS4","MS5","MS6","MS7","MS8","MS9","MS10","MS11"],
}

@dataclass
class AnnotatedSegment:
    labels: List[str]
    segment_length: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    chunk_index: Optional[int] = None
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotatedSegment':
        return cls(
            labels=data.get("labels", []),
            segment_length=data.get("segment_length", 0),
            start_index=data.get("start_index"),
            end_index=data.get("end_index"),
            chunk_index=data.get("chunk_index"),
            telemetry_data=data.get("telemetry_data", []),
            notes=data.get("notes")
        )

@dataclass
class PredictedSegment:
    labels: List[str]
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    start_index: Optional[int] = None
    end_index: Optional[int] = None

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
        features.extend(TelemetryFeatures.get_features_for_learning_expert())
        features.extend([f.value for f in ExpertFeatureCatalog.ExpertFeatures])
        features.extend([f.value for f in TireGripFeatureCatalog.ContextFeature])

        # Use dict.fromkeys to remove duplicates while preserving insertion order
        return list(dict.fromkeys(features))

