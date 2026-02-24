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
    "RM": "Recovery & Merge",
    "MS" :"Mistake segment",
    ################### Detailed Expert Adherence Labels (for label EA) ###################

    ################### Detailed Recovery & Merge Labels (for label RM) ###################
    "RM1": "Recover from off-track",
    "RM2": "Exit pit lane",
    "RM5": "Recover from large speed gap",
    "RM6": "Recover from small speed gap",
    "RM7": "Merge back to expert line",
    ################### Detailed mistake labels (for label MS) ###################
    "MS1": "Brake too late",
    "MS2": "Entry too late",
    "MS3": "Apex too late",
    "MS4": "Exit out too early",
    "MS5": "Brake too early",
    "MS6": "Entry too early",
    "MS7": "Apex too early",
    "MS8": "Exit out too late",
    "MS9": "Entry too tight",
    "MS10": "Apex too wide",
    "MS11": "Exit too narrow",
    "MS12": "Missing Apex",
    "MS13": "Brake not enough",
    "MS14" : "Brake applied too quickly",
    "MS15" : "Throttle applied too quickly",
    "MS16" : "Exit too wide",
    "MS17": "Release brake too quickly",
    "MS18": "Release brake too slowly",
    "MS19": "Throttle not enough",
    "MS20":"Throttle too early",
    "MS21":"Throttle too late",
    "MS22":"Brake too much",
    "MS23":"Release throttle too quickly",
    "MS24": "Brake applied too slowly",
    "MS25": "Throttle applied too slowly",
    "MS26": "Throttle too much",
    "MS27": "Release brake too late",
    "MS28": "Release brake too early",
    "MS29": "Release throttle too late",
    "MS30": "Release throttle too early",
     ################### Corner name ###################
    "brands_hatch":"Brands Hatch",
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

    ################### Other Labels ###################
    "Other1": "In the corner",
    "Other2": "On the straight",
    "Other3": "Approach to corner",
    "Other4": "Exit corner",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

MAIN_LABEL_GUIDELINES = {
    "1": "Overtaking: Analyze the driver's attempt to pass an opponent. Look for late braking, line deviation to find a gap, and speed differentials. Assess if the move was successful and safe.",
    "2": "Missing Data: This segment contains gaps or corrupted telemetry. Identify if the sensor data drops to zero or becomes inconsistent unexpectedly.",
    "EA": "Expert Adherence: The driver is following the optimal racing line and speed profile closely. nothing needs to be labeled right now.",
    "4": "Pit Stop: The driver is entering, waiting in, or exiting the pit lane. Look for significant speed reduction and distinct trajectory deviation into the pit area.",
    "RM": "Recovery & Merge: The driver is recovering from mistake such as slower speed. identify if driver is recovery from low speed, off-track, or merge back to expert line.",
    "MS": "Mistake Segment: The driver has committed a driving error resulting in time loss or instability. \n   - Step 1: Analyze the 'Time Difference' graph to pinpoint where the driver starts losing time compared to the expert.\n   - Step 2: Examine the 'Throttle', 'Brake', and 'Speed' traces at that specific moment.\n   - Step 3: Check the 'Trajectory' map for deviations from the optimal line.\n   - Step 4: Identify the *root cause* (e.g., braking too late, turning in too early, throttle applied too quickly) vs. consequential errors.",
    "brands_hatch": "Circuit Feature (Brands Hatch): Identify specific named corners or straight sections of the Brands Hatch circuit based on the trajectory shape and position. Also, identify the shape of the segment using labels other1-4 (e.g. in the corner, on the straight, approach to corner, exit corner)."
}

LABEL_CATEGORIES = {
    "Main Labels": ["1", "2","EA","4","RM","MS","brands_hatch"],
    "1":[],
    "2":[],
    "EA":[],
    "4":[],
    "RM":["RM1", "RM2", "RM5","RM6","RM7"],
    "brands_hatch":["brands_hatch1","brands_hatch2","brands_hatch3","brands_hatch4","brands_hatch5","brands_hatch6","brands_hatch7","brands_hatch8","brands_hatch9","brands_hatch10","brands_hatch11","brands_hatch12","brands_hatch13","brands_hatch14","brands_hatch15","brands_hatch16"],
    "MS":["MS1","MS2","MS3","MS4","MS5","MS6","MS7","MS8","MS9","MS10","MS11","MS12","MS13","MS14","MS15","MS16","MS17","MS18","MS19","MS20","MS21","MS22","MS23","MS24","MS25","MS26","MS27","MS28","MS29","MS30"],
    "Other Labels": ["Other1", "Other2", "Other3", "Other4"]
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

