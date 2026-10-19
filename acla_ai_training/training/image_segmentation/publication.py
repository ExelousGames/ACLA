"""Export segmentation checkpoints as ONNX for the backend model API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.model_publication import upload_ultralytics_model


async def upload_checkpoint(
    checkpoint: Path,
    *,
    name: str = "track-segments",
    backend_service=None,
) -> dict[str, Any]:
    """Read checkpoint metadata, export ONNX, and upload only the exported graph."""
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("Model file must use the .pt extension")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint is missing or empty: {checkpoint}")
    if not name.strip() or len(name) > 200:
        raise ValueError("Model name must contain 1 to 200 characters")

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    if model.task != "segment":
        raise ValueError("Only segmentation checkpoints can be published by this component")
    names = model.names
    args = model.ckpt.get("train_args", {})
    metadata = {
        "name": name.strip(),
        "task": "segment",
        "classNames": [names[index] for index in range(len(names))],
        "metadata": {
            "artifactFormat": "onnx",
            "trainingRunId": checkpoint.parent.parent.name,
            "baseModel": args.get("model"),
            "epochs": args.get("epochs"),
            "imgsz": args.get("imgsz"),
            "batch": args.get("batch"),
            "metrics": model.ckpt.get("train_metrics", {}),
        },
    }
    # Reject invalid JSON metrics before exporting or transferring the model.
    json.dumps(metadata, allow_nan=False)
    artifact = Path(model.export(
        format="onnx", imgsz=args.get("imgsz", 768), batch=1, dynamic=False,
        device="cpu", opset=17, simplify=False, nms=None,
    ))
    if artifact.suffix.lower() != ".onnx" or not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(f"ONNX export is missing or empty: {artifact}")
    del model

    if backend_service is None:
        from app.integrations.backend.client import backend_service

    with artifact.open("rb") as model_file:
        return await upload_ultralytics_model(
            backend_service, model_file, artifact.name, metadata,
        )
