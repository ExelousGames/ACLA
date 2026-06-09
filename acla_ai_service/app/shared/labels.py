"""Behavioural label catalog (ID → display name) and grouping metadata.

Pure data: dicts indexed by short label ID. No behaviour, no imports
from any framework or service. Moved from app/models/segment_models.py
in refactor/hexagonal-v1, Step 3.
"""

from typing import Any, Dict, List


LEGACY_LABEL_MAP: Dict[Any, str] = {
    # Original integer-coded main labels.
    "3": "EA",
    "5": "RM",
    3: "EA",
    5: "RM",
    "1": "O",
    "2": "MD",
    "4": "PS",

    # Mistake split (Practice vs Racing).
    "MS": "MSP",
    "MS1": "MSP1",
    "MS2": "MSP2",
    "MS3": "MSP3",
    "MS4": "MSP4",
    "MS5": "MSP5",
    "MS6": "MSP6",
    "MS7": "MSP7",
    "MS8": "MSP8",
    "MS9": "MSP9",
    "MS10": "MSP10",
    "MS11": "MSP11",
    "MS13": "MSP13",
    "MS14": "MSP14",
    "MS15": "MSP15",
    "MS16": "MSP16",
    "MS17": "MSP17",
    "MS18": "MSP18",
    "MS19": "MSP19",
    "MS20": "MSP20",
    "MS21": "MSP21",
    "MS22": "MSP22",
    "MS23": "MSP23",
    "MS24": "MSP24",
    "MS25": "MSP25",
    "MS26": "MSP26",
    "MS27": "MSP27",
    "MS28": "MSP28",
    "MS29": "MSP29",
    "MS30": "MSP30",
    "MS31": "MSP31",
    "MS32": "MSP32",
    "MS33": "MSP33",
    "MS34": "MSP34",
    "MS35": "MSP35",
    "MS36": "MSP36",
    "MS37": "MSP37",
    "MS38": "MSP38",
    "MS41": "MSP41",
    "MS42": "MSP42",
    "MS43": "MSP43",
    "MS44": "MSP44",
    "MS45": "MSP45",
    "MS46": "MSP46",
    "MS47": "MSP47",
    "MS48": "MSP48",
    "MS49": "MSP49",
    "MS50": "MSP50",
    "MS51": "MSP51",
    "MS52": "MSP52",
    "MS53": "MSR1",
    "MS54": "MSR2",

    # Overtaking split (offensive O vs defensive OD).
    "O2": "OD1",
    "O6": "OD2",
}


def normalize_label_id(label: Any) -> str:
    return str(LEGACY_LABEL_MAP.get(label, LEGACY_LABEL_MAP.get(str(label), label)))


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
    "MD": "Missing data",
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
    ################### Detailed Successful Defense Labels (for label OD) ###################
    "OD1": "Inside cover (early-brake defense)",
    "OD2": "Defensive lift on straight",
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
    ################### Detailed Practice-Mistake Labels (for label MSP) — technical errors, opponent-agnostic ###################
    "MSP1": "Initiate brake too late",
    "MSP2": "Initiate the turn too late",
    "MSP3": "Too late compared to expert Apex",
    "MSP4": "time to exit too early",
    "MSP5": "Initiate brake too early",
    "MSP6": "Initiate the turn too early",
    "MSP7": "Too early compared to expert Apex",
    "MSP8": "time to exit too late",
    "MSP9": "Entry trajectory too tight",
    "MSP10": "Apex too wide",
    "MSP11": "Exit trajectory too narrow",
    "MSP13": "Highest Brake pressure too low",
    "MSP14": "Brake applied too quickly",
    "MSP15": "Throttle applied too quickly",
    "MSP16": "Exit trajectory too wide",
    "MSP17": "Release brake too quickly",
    "MSP18": "Release brake too slowly",
    "MSP19": "Highest throttle pressure too low",
    "MSP20": "Initiate throttle too early",
    "MSP21": "Initiate throttle too late",
    "MSP22": "Highest Brake pressure too high",
    "MSP23": "Release throttle too quickly",
    "MSP24": "Brake applied too slowly",
    "MSP25": "Throttle applied too slowly",
    "MSP26": "Highest throttle pressure too high",
    "MSP27": "Initiate brake release too late",
    "MSP28": "Initiate brake release too early",
    "MSP29": "Initiate throttle release too late",
    "MSP30": "Initiate throttle release too early",
    "MSP31": "Highest Brake length too short",
    "MSP32": "Highest Brake length too long",
    "MSP33": "Entry trajectory too wide",
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
    ################### Detailed Racing-Mistake Labels (for label MSR) — interaction-failure with a close opponent ###################
    "MSR1": "Failed overtake attempt (type unclear)",
    "MSR2": "Defense broken (type unclear)",
    "MSR3": "Failed late-brake attack at entry",
    "MSR4": "Failed outside-line sweep",
    "MSR5": "Failed switchback",
    "MSR6": "Failed slipstream gain on straight",
    "MSR7": "Inside cover broken (early-brake defense)",
    "MSR8": "Defensive lift broken on straight",
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

    "moza":"Moza",
    "moza1":"Straight between 1 Varinate and Curva Parabolica",
    "moza2":"1 Varinate 1st turn",
    "moza3":"1 Varinate 2nd turn",
    "moza4": "Curva Biassono",
    "moza5": "2 Varinate",
    "moza6": "Curve di Lesmo first curve",
    "moza7": "Curve di Lesmo second curve",
    "moza8": "Curva del Serraglio",
    "moza9": "Variante Ascari 1st turn",
    "moza10": "Variante Ascari 2nd turn",
    "moza11": "Variante Ascari 3rd turn",
    "moza12": "Straight between Variante Ascari and Curva Parabolica",
    "moza13": "Curva Parabolica",

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

MAIN_LABEL_GUIDELINES: Dict[str, str] = {
    "O": "Successful Overtake: Player gained a position. Requires `classify_opponent_interaction` with `outcome: pass_completed`, `gates.O: true`, and `label_gates.O: true`. Without that deterministic pass outcome, do NOT attach O; a failed attack belongs in MSR.",
    "OD": "Successful Defense: Player held position against a real threat. Requires `classify_opponent_interaction` with `outcome: held_defense`, `gates.OD: true`, and `label_gates.OD: true`. A merely close opponent is not enough; no held-defense outcome means no OD.",
    "MD": "Missing Data: This segment contains gaps or corrupted telemetry. Identify if the sensor data drops to zero or becomes inconsistent unexpectedly.",
    "EA": "Expert Adherence (Training): The driver is following the optimal racing line and speed profile closely. nothing needs to be labeled right now.",
    "PS": "Pit Stop: The driver is entering, waiting in, or exiting the pit lane. Look for significant speed reduction and distinct trajectory deviation into the pit area.",
    "RM": "Recovery & Merge: The driver is recovering from mistake such as slower speed. identify if driver is recovery from low speed, off-track, or merge back to expert line.",
    "MSP": "Practice Mistake: A technical driving error (timing, line, brake/throttle modulation, gear, off-track, …) NOT caused by interaction with another car. \n   - Step 1: Analyze the 'Time Difference to Expert' graph; a mistake normally causes this time gap to increase continuously without decreasing shortly after.\n   - Step 2: Examine the differences in 'throttle' and 'brake' between the player and the expert to find out why the player is slower.\n   - Step 3: Examine the 'push_limit' graph to find understeer or oversteer, which could lead to lower speed.\n   - Step 4: Check the 'Trajectory' map for mistakes caused by deviations from the optimal line.\n   - Step 5: Find the MSP sub-label that describes the *root cause*.",
    "MSR": "Racing Mistake: Time / position loss caused by an interaction with a close opponent. Requires `classify_opponent_interaction` with `outcome: failed_attack` or `outcome: broken_defense`, `gates.MSR: true`, and `label_gates.MSR: true`. Failed overtakes mirror the O subtypes (use MSR3 for a failed O1 late-brake attack, MSR4 for a failed O3 outside-line sweep, MSR5 for a failed O4 switchback, MSR6 for a failed O5 slipstream gain). Broken defenses mirror OD (MSR7 for a broken OD1 inside cover, MSR8 for a broken OD2 defensive lift). Use the generic MSR1 / MSR2 only when the attempt type cannot be identified from telemetry. If no adverse classifier outcome is present, it's not MSR.",
    "brands_hatch": "Circuit Feature (Brands Hatch):Look at the Trajectories Overlay and Identify specific named corners or straight sections of the Brands Hatch circuit. Also, identify the shape of the segment using ST labels.",
    "silverstone": "Circuit Feature (Silverstone): Look at the Trajectories Overlay and Identify specific named corners or straight sections of the Silverstone circuit. Also, identify the shape of the segment using ST labels.",
    "Segment Type": "Segment Type: use ST1-ST6 for the base segment shape. For corner segments, add one corner-shape refinement from ST7-ST11 when clear, and add entry, apex, and exit altitude labels from ST12-ST20 when altitude evidence is clear."
}

LABEL_IMAGE_MAP: Dict[str, str] = {
    "brands_hatch": "Brands_Hatch_2003.jpg",
    "silverstone": "Silverstone_Circuit_2020.jpg"
}

LABEL_CATEGORIES: Dict[str, List[str]] = {
    "Main Labels": ["O", "OD", "MD","EA","PS","RM","MSP","MSR","brands_hatch","silverstone"],
    "O":["O1","O3","O4","O5"],
    "OD":["OD1","OD2"],
    "MD":[],
    "EA":[],
    "PS":[],
    "RM":["RM1", "RM2", "RM5","RM6","RM7","RM8","RM9","RM10","RM11"],
    "brands_hatch":["brands_hatch1","brands_hatch2","brands_hatch3","brands_hatch4","brands_hatch5","brands_hatch6","brands_hatch7","brands_hatch8","brands_hatch9","brands_hatch10","brands_hatch11","brands_hatch12","brands_hatch13","brands_hatch14","brands_hatch15","brands_hatch16","brands_hatch17","brands_hatch18","brands_hatch19"],
    "silverstone":["silverstone1","silverstone2","silverstone3","silverstone4","silverstone5","silverstone6","silverstone7","silverstone8","silverstone9","silverstone10","silverstone11","silverstone12","silverstone13","silverstone14","silverstone15","silverstone16","silverstone17","silverstone18","silverstone20","silverstone21","silverstone19"],
    "moza":["moza1","moza2","moza3","moza4","moza5","moza6","moza7","moza8","moza9","moza10","moza11","moza12","moza13"],
    "MSP":["MSP1","MSP2","MSP3","MSP4","MSP5","MSP6","MSP7","MSP8","MSP9","MSP10","MSP11","MSP13","MSP14","MSP15","MSP16","MSP17","MSP18","MSP19","MSP20","MSP21","MSP22","MSP23","MSP24","MSP25","MSP26","MSP27","MSP28","MSP29","MSP30","MSP31","MSP32","MSP33","MSP34","MSP35","MSP36","MSP37","MSP38","MSP41","MSP42","MSP43","MSP44","MSP45","MSP46","MSP47","MSP48","MSP49","MSP50","MSP51","MSP52"],
    "MSR":["MSR1","MSR2","MSR3","MSR4","MSR5","MSR6","MSR7","MSR8"],
    "Segment Type": ["ST1", "ST2", "ST3", "ST4", "ST5", "ST6", "ST7", "ST8", "ST9", "ST10", "ST11", "ST12", "ST13", "ST14", "ST15", "ST16", "ST17", "ST18", "ST19", "ST20"]
}


__all__ = [
    "LEGACY_LABEL_MAP",
    "LABEL_MAPPING",
    "LABEL_NAME_TO_ID",
    "MAIN_LABEL_GUIDELINES",
    "LABEL_IMAGE_MAP",
    "LABEL_CATEGORIES",
    "normalize_label_id",
    "normalize_label_ids",
]
