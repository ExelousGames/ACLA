"""Load prepared image, XYZ, and track-context inputs without reading driver telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


OUTPUT_NAMES = tuple(f"reference_delta_{axis}" for axis in "xyz")


def position_context(normalized_position, game, map_name):
    """Validate the context shared by preparation, training, and prediction."""
    if (
        isinstance(normalized_position, bool) or not isinstance(normalized_position, (int, float))
        or not np.isfinite(normalized_position) or not 0 <= normalized_position <= 1
    ):
        raise ValueError("normalized_position must be a finite number between 0 and 1.")
    context = {"normalized_position": float(normalized_position)}
    for field, value in (("game", game), ("map_name", map_name)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty name.")
        context[field] = value.strip()
    return context


def encode_position_context(normalized_position, game, map_name, categories):
    """Encode lap progress followed by game and map one-hot vectors in saved order."""
    context = position_context(normalized_position, game, map_name)
    encoded = [context["normalized_position"]]
    for field in ("game", "map_name"):
        names = categories[field]
        if context[field] not in names:
            raise ValueError(f"Unknown {field}: {context[field]!r}; use a model trained with this name.")
        encoded.extend(float(name == context[field]) for name in names)
    return np.asarray(encoded, dtype=np.float32)


class PositionCurve:
    """A closed-lap XYZ curve indexed by normalized lap position."""

    def __init__(self, positions, xyz):
        positions = np.asarray(positions, dtype=np.float64)
        xyz = np.asarray(xyz, dtype=np.float64)
        if positions.ndim != 1 or len(positions) < 2 or xyz.shape != (len(positions), 3):
            raise ValueError("A position curve needs at least two normalized positions and matching XYZ triples.")
        if not np.isfinite(positions).all() or not np.isfinite(xyz).all():
            raise ValueError("Position curves must contain finite values.")
        if np.any((positions < 0) | (positions > 1)):
            raise ValueError("Curve normalized positions must be between 0 and 1.")
        order = np.argsort(positions)
        positions, xyz = positions[order], xyz[order]
        if np.any(np.diff(positions) <= 0):
            raise ValueError("Curve normalized positions must be unique.")
        # Preserve explicit 0/1 endpoints; otherwise interpolate across the seam.
        self.positions, self.xyz = positions, xyz
        if positions[0] > 0:
            self.positions = np.concatenate(([positions[-1] - 1], self.positions))
            self.xyz = np.concatenate((xyz[-1:], self.xyz))
        if positions[-1] < 1:
            self.positions = np.concatenate((self.positions, [positions[0] + 1]))
            self.xyz = np.concatenate((self.xyz, xyz[:1]))

    @classmethod
    def from_points(cls, points):
        return cls([p["normalized_position"] for p in points], [[p[a] for a in "xyz"] for p in points])

    @classmethod
    def from_top_lap(cls, entry):
        from training.top_laps.reference import ExpertFeatureCatalog

        features = ExpertFeatureCatalog.ExpertOptimalFeature
        names = [getattr(features, f"EXPERT_OPTIMAL_PLAYER_POS_{a.upper()}").value for a in "xyz"]
        if not all(name in entry.target_features for name in names):
            raise ValueError("The selected reference lap does not contain all three player XYZ coordinates.")
        return cls(entry.x, entry.y[:, [entry.target_features.index(name) for name in names]])

    def at(self, normalized_position):
        position = float(normalized_position)
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("Normalized position must be finite and between 0 and 1.")
        return np.array([np.interp(position, self.positions, self.xyz[:, axis]) for axis in range(3)])


def reference_inputs(normalized_position, reference_lap: PositionCurve, centerline: PositionCurve):
    """Reference XYZ inputs, with no driver coordinates or derived deltas."""
    return np.stack((reference_lap.at(normalized_position), centerline.at(normalized_position)))


def load_position_manifest(path: Path) -> dict[str, list[dict]]:
    """Read a position-file folder or dataset index, grouped by session without splitting.

    Each frame contains session_id, image, game, map_name, normalized_position, references,
    and target. Arrays contain reference-lap XYZ and optionally centerline XYZ.
    Image paths are relative to their JSON file. No telemetry or curve files are read.
    """
    path = path.resolve()
    sessions, images = {}, set()
    if path.is_dir():
        position_files = [(None, file) for file in sorted(path.rglob("*.position.json"))]
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        position_files = []
        for session in manifest["sessions"]:
            session_id = session["id"]
            if not isinstance(session_id, str) or not session_id or session_id in sessions:
                raise ValueError("Each session must have a unique, nonempty id.")
            if "split" in session:
                raise ValueError("Preset session splits are unsupported; rerun prepare_position_dataset.py.")
            sessions[session_id] = []
            for frame in session["frames"]:
                if set(frame) != {"position_file"}:
                    raise ValueError("Index frames require only position_file; rerun prepare_position_dataset.py.")
                position_files.append((session_id, (path.parent / frame["position_file"]).resolve()))

    for expected_session, position_path in position_files:
        frame = json.loads(position_path.read_text(encoding="utf-8"))
        session_id = frame["session_id"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Each position file must have a nonempty session_id.")
        if expected_session is not None and session_id != expected_session:
            raise ValueError(f"Position file belongs to another session: {position_path}")
        if "references" not in frame or "target" not in frame:
            raise ValueError("Position files require references and target arrays; rerun prepare_position_dataset.py.")
        if not {"game", "map_name", "normalized_position"} <= frame.keys():
            raise ValueError(
                "Position files require game, map_name, and normalized_position; "
                "regenerate the dataset with prepare_position_dataset.py."
            )
        image = (position_path.parent / frame["image"]).resolve()
        if image in images:
            raise ValueError(f"Image appears more than once in the position dataset: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"Image does not exist: {image}")
        images.add(image)
        record = {
            "session_id": session_id, "image": image,
            **position_context(frame["normalized_position"], frame["game"], frame["map_name"]),
        }
        for field in ("references", "target"):
            values = np.asarray(frame[field], dtype=np.float32)
            if values.shape not in ((1, 3), (2, 3)) or not np.isfinite(values).all():
                raise ValueError(f"Frame {field} must contain one or two finite XYZ triples: {image}")
            record[field] = values
        sessions.setdefault(session_id, []).append(record)
    if not sessions or not all(sessions.values()):
        raise ValueError("Position datasets require nonempty sessions with prepared frames.")
    return sessions


def image_tensor(path: Path, imgsz: int) -> torch.Tensor:
    """Letterbox the complete RGB frame without changing its camera geometry."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    ratio = min(imgsz / image.width, imgsz / image.height)
    image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))))
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(image, ((imgsz - image.width) // 2, (imgsz - image.height) // 2))
    return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255


class PositionDataset(Dataset):
    def __init__(self, records, imgsz, categories):
        self.records, self.imgsz = records, imgsz
        self.categories = categories

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        # Use only the reference lap; prepared files may also contain centerline XYZ.
        # Keep inputs and targets separate. No raw telemetry is returned to the trainer.
        return {
            "game": row["game"],
            "map_name": row["map_name"],
            "image": image_tensor(row["image"], self.imgsz),
            "references": torch.from_numpy(row["references"][:1]),
            "context": torch.from_numpy(encode_position_context(
                row["normalized_position"], row["game"], row["map_name"], self.categories,
            )),
            "target": torch.from_numpy(row["target"][:1]),
        }
