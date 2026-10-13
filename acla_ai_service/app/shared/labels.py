"""Behavioural label catalog (ID → display name) and grouping metadata.

Pure data: dicts indexed by short label ID. No behaviour, no imports
from any framework or service. Moved from app/models/segment_models.py
in refactor/hexagonal-v1, Step 3.
"""

from typing import Any, Dict, List


def normalize_label_id(label: Any) -> str:
    return str(label)


def normalize_label_ids(labels: Any) -> List[str]:
    if not isinstance(labels, list):
        return []

    normalized: List[str] = []
    seen = set()
    for label in labels:
        current = normalize_label_id(label)
        if current not in seen:
            normalized.append(current)
            seen.add(current)
    return normalized


LABEL_MAPPING: Dict[str, str] = {
    ################### Main Labels ###################
    "O": "Successful Overtake",
    "OD": "Successful Defense",
    "EA": "Expert Adherence (Training)",
    "PS": "Pit Stop",
    
    "RM": "Recovery & Merge",
    "MSP": "Mistake (Practice)",
    "MSR": "Mistake (Racing)",
    ################### Detailed Expert Adherence (Training) Labels (for label EA) ###################

    ################### Detailed Successful Overtake Labels (for label O) ###################
    "O1": "Late-brake attack at entry",
    "O3": "Outside-line sweep",
    "O4": "Switchback",
    "O5": "Slipstream gain on straight",

    ################### Detailed Pit Stop Labels (for label PS) ###################
    "PS1": "Exit pit lane",
    
    ################### Detailed Successful Defense Labels (for label OD) ###################
    "OD1": "Inside cover (early-brake defense)",
    "OD2": "Defensive lift on straight",
    ################### Detailed Recovery & Merge Labels (for label RM) ###################
    "RM1": "Recover from off-track",
    "RM5": "Recover from speed gap over 20",
    "RM6": "Recover from small speed gap under 20",
    "RM7": "Merge back to expert line",
    "RM8": "Accelerate eariler at exit",
    "RM9": "Accelerate later at exit",
    "RM10": "Brake earlier at entry",
    "RM11": "Brake later at entry",
    ################### Detailed Practice-Mistake Labels (for label MSP) — technical errors, opponent-agnostic ###################
    "MSP1": "Initiate brake too late",
    "MSP2": "Initiate the turn too late",
    "MSP3": "Too late compared to expert Apex",
    "MSP4": "Initiate exit too early",
    "MSP5": "Initiate brake too early",
    "MSP6": "Initiate the turn too early",
    "MSP7": "Too early compared to expert Apex",
    "MSP8": "Initiate exit too late",
    "MSP9": "Entry trajectory started too tight",
    "MSP10": "Apex too wide",
    "MSP11": "Exit trajectory started too narrow",
    "MSP13": "Highest Brake pressure too low",
    "MSP14": "Brake applied too quickly",
    "MSP15": "Throttle reapplied too quickly",
    "MSP16": "Exit trajectory started too wide",
    "MSP17": "Release brake too quickly",
    "MSP18": "Release brake too slowly",
    "MSP19": "Brake not hard enough",
    "MSP20": "Reapply throttle too early",
    "MSP21": "Reapply throttle too late",
    "MSP22": "Hard brake too long",
    "MSP23": "Release throttle too quickly",
    "MSP24": "Brake applied too slowly",
    "MSP25": "Throttle reapplied too slowly",
    "MSP26": "Highest throttle pressure too high",
    "MSP27": "Initiate brake release too late",
    "MSP28": "Initiate brake release too early",
    "MSP29": "Initiate throttle release too late",
    "MSP30": "Initiate throttle release too early",
    "MSP31": "Highest Brake length too short",
    "MSP32": "Highest Brake length too long",
    "MSP33": "Entry trajectory started too wide",
    "MSP34": "Throttle and brake applied at the same time for too long",
    "MSP35": "Shift up too early",
    "MSP36": "Shift up too late",
    "MSP37": "Shift down too early",
    "MSP38": "Shift down too late",
    "MSP41": "Cutting the line",
    "MSP42": "Inefficient grip utilization at entry",
    "MSP43": "Inefficient grip utilization at exit",
    "MSP44": "Oversteering at entry",
    "MSP45": "Understeering at entry",
    "MSP46": "Oversteering at exit",
    "MSP47": "Understeering at exit",
    "MSP48": "Gear Too low when accelerating",
    "MSP49": "Gear Too high when accelerating",
    "MSP50": "Off track at entry",
    "MSP51": "Off track at exit",
    "MSP52": "Off track in the straight",
    "MSP53": "Driver cruising",
    "MSP54": "Gear too low when braking",
    "MSP55": "Gear too high when braking",
    ################### Detailed Racing-Mistake Labels (for label MSR) — interaction-failure with a close opponent ###################
    "MSR1": "Failed overtake attempt (type unclear)",
    "MSR2": "Defense broken (type unclear)",
    "MSR3": "Failed late-brake attack at entry",
    "MSR4": "Failed outside-line sweep",
    "MSR5": "Failed switchback",
    "MSR6": "Failed slipstream gain on straight",
    "MSR7": "Inside cover broken (early-brake defense)",
    "MSR8": "Defensive lift broken on straight",
    ################### Segment Type ###################
    "ST1": "In the corner",
    "ST2": "On the straight",
    "ST3": "Approach to corner",
    "ST4": "Exit corner leading to straight",
    "ST5": "Between consecutive corners",
    "ST6": "Consecutive corners with no straight in between",
    "ST7": "Constant-radius corner",
    "ST8": "Increasing-radius corner",
    "ST9": "Decreasing-radius corner",
    "ST10": "Hairpin corner",
    "ST11": "Chicane or esses",
    "ST12": "Entry altitude uphill",
    "ST13": "Entry altitude level",
    "ST14": "Entry altitude downhill",
    "ST15": "Apex altitude uphill",
    "ST16": "Apex altitude level",
    "ST17": "Apex altitude downhill",
    "ST18": "Exit altitude uphill",
    "ST19": "Exit altitude level",
    "ST20": "Exit altitude downhill",
}
LABEL_NAME_TO_ID: Dict[str, str] = {v: k for k, v in LABEL_MAPPING.items()}

LABEL_IMAGE_MAP: Dict[str, str] = {}

BEHAVIOR_LABELS: List[str] = ["O", "OD", "EA", "PS", "RM", "MSP", "MSR"]

LABEL_CATEGORIES: Dict[str, List[str]] = {
    "Main Labels": BEHAVIOR_LABELS,
    "Behavior": BEHAVIOR_LABELS,
    "O":["O1","O3","O4","O5"],
    "OD":["OD1","OD2"],
    "EA":[],
    "PS":["PS1"],
    "RM":["RM1", "RM5","RM6","RM7","RM8","RM9","RM10","RM11"],
    "MSP":["MSP1","MSP2","MSP3","MSP4","MSP5","MSP6","MSP7","MSP8","MSP9","MSP10","MSP11","MSP13","MSP14","MSP15","MSP16","MSP17","MSP18","MSP19","MSP20","MSP21","MSP22","MSP23","MSP24","MSP25","MSP26","MSP27","MSP28","MSP29","MSP30","MSP31","MSP32","MSP33","MSP34","MSP35","MSP36","MSP37","MSP38","MSP41","MSP42","MSP43","MSP44","MSP45","MSP46","MSP47","MSP48","MSP49","MSP50","MSP51","MSP52","MSP53","MSP54","MSP55"],
    "MSR":["MSR1","MSR2","MSR3","MSR4","MSR5","MSR6","MSR7","MSR8"],
    "Segment Type": ["ST1", "ST2", "ST3", "ST4", "ST5", "ST6", "ST7", "ST8", "ST9", "ST10", "ST11", "ST12", "ST13", "ST14", "ST15", "ST16", "ST17", "ST18", "ST19", "ST20"]
}


__all__ = [
    "LABEL_MAPPING",
    "LABEL_NAME_TO_ID",
    "LABEL_IMAGE_MAP",
    "BEHAVIOR_LABELS",
    "LABEL_CATEGORIES",
    "normalize_label_id",
    "normalize_label_ids",
]
