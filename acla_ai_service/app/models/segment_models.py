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
    "EA": "Expert Adherence",
    "4": "Pit Stop",
    "5": "Recovery & Merge",
    "MS" :"Mistake segment",
    ################### Detailed Expert Adherence Labels (for label EA) ###################
    "EA1": "",
   
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
     ################### Corner name ###################
    "brands_hatch1":"Brabham Straight",
    "brands_hatch2":"Paddock Hill Bend",
    "brands_hatch3":"Druids",
    "brands_hatch4":"Graham Hill Bend",
    "brands_hatch5":"Cooper Straight",
    "brands_hatch6":"Surtees",
    "brands_hatch7":"Pilgrim's Drop",
    "brands_hatch8":"Hawthorn Hill",     
    "brands_hatch9":"Hawthorn Bend",
    "brands_hatch10":"Derek Minter Straight",
    "brands_hatch11":"Westfield Bend",
    "brands_hatch12":"Dingle Dell",
    "brands_hatch13":"Sheene Curve",
    "brands_hatch14":"Stirling's Bend",
    "brands_hatch15":"Clearways",
    "brands_hatch16":"Clark Curve",  
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

LABEL_CATEGORIES = {
    "Main Labels": ["1", "2","EA","4","5","MS"],
    "1":[],
    "2":[],
    "EA":[],
    "4":[],
    "5":[],
    "brands_hatch":["brands_hatch1","brands_hatch2","brands_hatch3","brands_hatch4","brands_hatch5","brands_hatch6","brands_hatch7","brands_hatch8","brands_hatch9","brands_hatch10","brands_hatch11","brands_hatch12","brands_hatch13","brands_hatch14","brands_hatch15","brands_hatch16"],
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

