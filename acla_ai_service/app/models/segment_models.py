from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Constants
LABEL_MAPPING = {
    1: "Overtaking",
    2: "Tire Strategy",
    3: "Expert Line Adherence",
    4: "Mistake - Brake too early",
    5: "Recovery",
    6: "Mistake - Brake too late",
    7: "Mistake - Brake too much",
    8: "Mistake - Brake too little",
    9: "Mistake - Accelerate too early",
    10: "Mistake - Accelerate too late",
    11: "Mistake - Accelerate too little",
    12: "Mistake - Accelerate too much",
    13: "Expert Speed Adherence",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_MAPPING.items()}

@dataclass
class AnnotatedSegment:
    labels: List[int]
    segment_length: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    chunk_index: Optional[int] = None
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)

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
            telemetry_data=data.get("telemetry_data", [])
        )

@dataclass
class PredictedSegment:
    labels: List[str]
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
