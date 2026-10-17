"""Regenerate with Python + NumPy; matches TopLapEntry.to_components()."""
import base64
import json
from pathlib import Path
import pickle

import numpy as np


entry = dict(
    game="acc", track="spa", car="car-a", avg_grip_int=2,
    x=np.array([0.0, 0.5, 1.0], dtype=float),
    y=np.array([[10.0, 1.0, 20.0, 100.0], [30.0, 2.0, 40.0, 150.0], [10.0, 1.0, 20.0, 100.0]]),
    target_features=["expert_optimal_player_pos_x", "expert_optimal_player_pos_y", "expert_optimal_player_pos_z", "expert_optimal_speed"],
)


def encode(value, protocol=5):
    return base64.b64encode(pickle.dumps(value, protocol=protocol)).decode("ascii")


fixtures = {"expected": {**entry, "x": entry["x"].tolist(), "y": entry["y"].tolist()}}
for protocol in (4, 5):
    for order in ("C", "F"):
        fixtures[f"protocol{protocol}{order}"] = encode({**entry, "y": np.array(entry["y"], order=order)}, protocol)
fixtures["bigEndian"] = encode({**entry, "x": entry["x"].astype(">f8"), "y": entry["y"].astype(">f8")})
fixtures["legacy"] = encode({key: value for key, value in entry.items() if key != "game"})
fixtures["iracing"] = encode({**entry, "game": "iracing", "track": "spa - layout", "car": "GT3", "avg_grip_int": 4})
fixtures["noCoordinates"] = encode({**entry, "y": np.empty((3, 0)), "target_features": []})
fixtures["invalidPositions"] = encode({**entry, "x": np.array([0.0, 0.5, 0.5])})
fixtures["nonFinite"] = encode({**entry, "y": np.full((3, 4), np.nan)})
fixtures["forbiddenGlobal"] = encode(eval)  # serialized only, never invoked
Path(__file__).with_name("top-lap-reference.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")
