import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from app.models.telemetry_models import TelemetryFeatures
from app.services.imitate_expert_learning_service import ExpertFeatureCatalog
from app.services.tire_grip_analysis_service import TireGripFeatureCatalog

# Constants
LABEL_MAPPING = {
    ################### Main Labels ###################
    "O": "Overtaking",
    "MD": "Missing data",
    "EA": "Expert Adherence",
    "PS": "Pit Stop",
    "RM": "Recovery & Merge",
    "MS" :"Mistake",
    ################### Detailed Expert Adherence Labels (for label EA) ###################

    ################### Detailed Recovery & Merge Labels (for label RM) ###################
    "RM1": "Recover from off-track",
    "RM2": "Exit pit lane",
    "RM5": "Recover from speed gap over 20",
    "RM6": "Recover from small speed gap under 20",
    "RM7": "Merge back to expert line",
    "RM8": "Accelerate eariler at exit",
    "RM9": "Accelerate later at exit",
    "RM10": "Brake earlier at entry",
    "RM11": "Brake later at entry",
    ################### Detailed mistake labels (for label MS) ###################
    "MS1": "Initiate brake too late",
    "MS2": "Initiate the turn too late",
    "MS3": "Too late compared to expert Apex",
    "MS4": "time to exit too early",
    "MS5": "Initiate brake too early",
    "MS6": "Initiate the turn too early",
    "MS7": "Too early compared to expert Apex",
    "MS8": "time to exit too late",
    "MS9": "Entry trajectory too tight",
    "MS10": "Apex too wide",
    "MS11": "Exit trajectory too narrow",
    "MS13": "Highest Brake pressure too low",
    "MS14" : "Brake applied too quickly",
    "MS15" : "Throttle applied too quickly",
    "MS16" : "Exit trajectory too wide",
    "MS17": "Release brake too quickly",
    "MS18": "Release brake too slowly",
    "MS19": "Highest throttle pressure too low",
    "MS20":"Initiate throttle too early",
    "MS21":"Initiate throttle too late",
    "MS22":"Highest Brake pressure too high",
    "MS23":"Release throttle too quickly",
    "MS24": "Brake applied too slowly",
    "MS25": "Throttle applied too slowly",
    "MS26": "Highest throttle pressure too high",
    "MS27": "Initiate brake release too late",
    "MS28": "Initiate brake release too early",
    "MS29": "Initiate throttle release too late",
    "MS30": "Initiate throttle release too early",
    "MS31": "Highest Brake length too short",
    "MS32": "Highest Brake length too long",
    "MS33":"Entry trajectory too wide",
    "MS34":"Throttle and brake applied at the same time for too long",
    "MS35": "Shift up too early",
    "MS36": "Shift up too late",
    "MS37": "Shift down too early",
    "MS38": "Shift down too late",
    "MS41": "Cutting the line",
    "MS42": "Inefficient grip utilization at entry",
    "MS43": "Inefficient grip utilization at exit",
    "MS44": "Oversteering at entry",
    "MS45": "Understeering at entry",
    "MS46": "Oversteering at exit",
    "MS47": "Understeering at exit",
    "MS48": "Gear Too low when accelerating",
    "MS49": "Gear Too high when accelerating",
    "MS50": "Off track at entry",
    "MS51": "Off track at exit",
    "MS52": "Off track in the straight",
              ################### Corner name ###################
    "brands_hatch":"Brands Hatch",
    "brands_hatch1":"Brabham Straight",
    "brands_hatch2":"Paddock Hill Bend",
    "brands_hatch3":"Druids",
    "brands_hatch4":"Graham Hill Bend",
    "brands_hatch5":"Cooper Straight",
    "brands_hatch6":"Surtees",
    "brands_hatch7":"Pilgrim's Drop",
    "brands_hatch9":"Hawthorns",
    "brands_hatch10":"Derek Minter Straight",
    "brands_hatch11":"Westfield Bend",
    "brands_hatch12":"Dingle Dell",
    "brands_hatch13":"Sheene Curve",
    "brands_hatch14":"Stirlings",
    "brands_hatch15":"Clearways",
    "brands_hatch16":"Clark Curve",  
    "brands_hatch17":"Pit",
    "brands_hatch18":"Graham Hill",
    "brands_hatch19":"Hailwoods Hill",

    "silverstone":"Silverstone",
    "silverstone1":"Woodcote",
    "silverstone2":"Copse",
    "silverstone3":"Maggotts",
    "silverstone4":"Becketts",
    "silverstone5":"Chapel",
    "silverstone6":"Hangar Straight",
    "silverstone7":"Stowe",
    "silverstone8":"Vale",
    "silverstone9":"Club",
    "silverstone10":"Hamilton Straight",
    "silverstone11":"Abbey",
    "silverstone12":"Farm Curve",
    "silverstone13":"Village",
    "silverstone14":"The Loop",
    "silverstone15":"Aintree",
    "silverstone16":"Wellington Straight",
    "silverstone17":"Brooklands",
    "silverstone18":"Luffield",
    "silverstone19":"Pit",
    "silverstone20":"Vale first 90 shape turn",
    "silverstone21":"Vale second over 90 rounded turn",

    ################### Segment Type ###################
    "ST1": "In the corner",
    "ST2": "On the straight",
    "ST3": "Approach to corner",
    "ST4": "Exit corner leading to straight",
    "ST5": "Between consecutive corners",
    "ST6": "Consecutive corners with no straight in between",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

MAIN_LABEL_GUIDELINES = {
    "O": "Overtaking: Analyze the driver's attempt to pass an opponent. Look for late braking, line deviation to find a gap, and speed differentials. Assess if the move was successful and safe.",
    "MD": "Missing Data: This segment contains gaps or corrupted telemetry. Identify if the sensor data drops to zero or becomes inconsistent unexpectedly.",
    "EA": "Expert Adherence: The driver is following the optimal racing line and speed profile closely. nothing needs to be labeled right now.",
    "PS": "Pit Stop: The driver is entering, waiting in, or exiting the pit lane. Look for significant speed reduction and distinct trajectory deviation into the pit area.",
    "RM": "Recovery & Merge: The driver is recovering from mistake such as slower speed. identify if driver is recovery from low speed, off-track, or merge back to expert line.",
    "MS": "Mistake Segment: The driver has committed a driving error resulting in time loss or instability. \n   - Step 1: Analyze the 'Time Difference to Expert' graph; a mistake normally causes this time gap to increase continuously without decreasing shortly after.\n   - Step 2: Examine the differences in 'throttle' and 'brake' between the player and the expert to find out why the player is slower.\n   - Step 3: Examine the 'push_limit' graph to find understeer or oversteer, which could lead to lower speed.\n   - Step 4: Check the 'Trajectory' map for mistakes caused by deviations from the optimal line.\n   - Step 5: Find the labels that describe the *root cause*.",
    "brands_hatch": "Circuit Feature (Brands Hatch):Look at the Trajectories Overlay and Identify specific named corners or straight sections of the Brands Hatch circuit. Also, identify the shape of the segment using labels ST1-6 ",
    "silverstone": "Circuit Feature (Silverstone): Look at the Trajectories Overlay and Identify specific named corners or straight sections of the Silverstone circuit. Also, identify the shape of the segment using labels ST1-6.",
    "Segment Type": "Segment Type: if the segment contains a corner, label the shape as in the corner. if the segment is on the straight, label as on the straight. if the segment is approaching a corner but not yet in it, label as approach to corner. if the segment is exiting a corner, label as exit corner. if the segment is between corners and hasnt be name yet, label as between corners."
}

LABEL_IMAGE_MAP = {
    "brands_hatch": "Brands_Hatch_2003.jpg",
    "silverstone": "Silverstone_Circuit_2020.jpg"
}

LABEL_CATEGORIES = {
    "Main Labels": ["O", "MD","EA","PS","RM","MS","brands_hatch","silverstone"],
    "O":[],
    "MD":[],
    "EA":[],
    "PS":[],
    "RM":["RM1", "RM2", "RM5","RM6","RM7","RM8","RM9","RM10","RM11"],
    "brands_hatch":["brands_hatch1","brands_hatch2","brands_hatch3","brands_hatch4","brands_hatch5","brands_hatch6","brands_hatch7","brands_hatch8","brands_hatch9","brands_hatch10","brands_hatch11","brands_hatch12","brands_hatch13","brands_hatch14","brands_hatch15","brands_hatch16","brands_hatch17","brands_hatch18","brands_hatch19"],
    "silverstone":["silverstone1","silverstone2","silverstone3","silverstone4","silverstone5","silverstone6","silverstone7","silverstone8","silverstone9","silverstone10","silverstone11","silverstone12","silverstone13","silverstone14","silverstone15","silverstone16","silverstone17","silverstone18","silverstone20","silverstone21","silverstone19"],
    "MS":["MS1","MS2","MS3","MS4","MS5","MS6","MS7","MS8","MS9","MS10","MS11","MS13","MS14","MS15","MS16","MS17","MS18","MS19","MS20","MS21","MS22","MS23","MS24","MS25","MS26","MS27","MS28","MS29","MS30","MS31","MS32","MS33","MS34","MS35","MS36","MS37","MS38","MS41","MS42","MS43","MS44","MS45","MS46","MS47","MS48","MS49","MS50","MS51","MS52"],
    "Segment Type": ["ST1", "ST2", "ST3", "ST4", "ST5", "ST6"]
}

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
            parent_id=data.get("parent_id")
        )

@dataclass
class PredictedSegment:
    labels: List[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
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

