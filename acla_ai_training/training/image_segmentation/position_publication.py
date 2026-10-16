"""Publish position checkpoints without treating them as segmentation weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from training.model_publication import upload_ultralytics_model

from .position_data import OUTPUT_NAMES


async def upload_position_checkpoint(
    checkpoint: Path,
    *,
    name: str = "track-position",
    backend_service=None,
) -> dict[str, Any]:
    """Read position metadata and stream the original checkpoint to the backend."""
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("Model file must use the .pt extension")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint is missing or empty: {checkpoint}")
    if not name.strip() or len(name) > 200:
        raise ValueError("Model name must contain 1 to 200 characters")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != "acla-yolo-position-v1":
        raise ValueError("Not an ACLA YOLO position checkpoint.")
    if saved.get("output_names") != list(OUTPUT_NAMES) or not saved.get("state_dict"):
        raise ValueError("Position checkpoint is missing its weights or signed XYZ outputs.")
    names = saved["names"]
    metadata = {
        "name": name.strip(),
        "task": "position",
        "classNames": [names[index] for index in range(len(names))],
        "metadata": {
            "format": saved["format"],
            "trainingRunId": checkpoint.parent.name,
            "imgsz": saved["imgsz"],
            "outputNames": saved["output_names"],
            "targetDefinition": saved["target_definition"],
            "coordinateUnits": saved["coordinate_units"],
            "metrics": saved.get("metrics", {}),
        },
    }
    json.dumps(metadata, allow_nan=False)
    del saved

    if backend_service is None:
        from app.integrations.backend.client import backend_service

    with checkpoint.open("rb") as model_file:
        return await upload_ultralytics_model(
            backend_service, model_file, checkpoint.name, metadata,
        )
