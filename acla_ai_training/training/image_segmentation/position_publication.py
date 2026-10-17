"""Publish browser-ready position ONNX artifacts through the model store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.model_publication import upload_ultralytics_model

from .position_export import METADATA_KEY, export_position_checkpoint


async def upload_position_checkpoint(
    checkpoint: Path,
    *,
    name: str = "track-position",
    backend_service=None,
) -> dict[str, Any]:
    """Export .pt checkpoints, or publish an already exported ONNX artifact."""
    import onnx

    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() not in (".pt", ".onnx"):
        raise ValueError("Model file must use the .pt or .onnx extension")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint is missing or empty: {checkpoint}")
    if not name.strip() or len(name) > 200:
        raise ValueError("Model name must contain 1 to 200 characters")

    if checkpoint.suffix.lower() == ".pt":
        checkpoint = export_position_checkpoint(checkpoint)
    graph = onnx.load(str(checkpoint))
    onnx.checker.check_model(graph)
    properties = {item.key: item.value for item in graph.metadata_props}
    if METADATA_KEY not in properties:
        raise ValueError("ONNX model is missing its Track Vision metadata.")
    metadata = json.loads(properties[METADATA_KEY])
    if metadata.get("task") != "position" or metadata.get("metadata", {}).get("artifactFormat") != "onnx":
        raise ValueError("ONNX model must describe a Track Vision position artifact.")
    metadata["name"] = name.strip()
    json.dumps(metadata, allow_nan=False)
    del graph

    if backend_service is None:
        from app.integrations.backend.client import backend_service

    with checkpoint.open("rb") as model_file:
        return await upload_ultralytics_model(
            backend_service, model_file, checkpoint.name, metadata,
        )
