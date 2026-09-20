"""Train image segmentation locally and publish the saved checkpoint to the backend."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import yaml

from . import WORKSPACE_DIR


def train_model(
    data: Path,
    *,
    model: str = "yolo11n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    workers: int = 0,
    project: Path = WORKSPACE_DIR / "storage/image_segmentation/runs",
    name: str = "train",
    upload: bool = True,
    upload_name: str = "track-segments",
):
    data = data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    from ultralytics import YOLO

    model_path = Path(model)
    if model_path.suffix == ".pt" and model_path.name == model and not model_path.is_file():
        # Give Ultralytics an absolute download target instead of the working directory.
        model = str(WORKSPACE_DIR / "storage/image_segmentation/pretrained" / model_path)
    network = YOLO(model, task="segment")
    if network.task != "segment":
        raise ValueError("Use a segmentation checkpoint or model YAML, such as yolo11n-seg.pt.")
    names = yaml.safe_load(data.read_text(encoding="utf-8")).get("names", [])
    if isinstance(names, dict):
        names = names.values()
    boundary_options = {}
    if {"left_boundary", "right_boundary"}.intersection(names):
        # Flips keep class IDs, incorrectly turning left boundaries into right.
        # Avoid further downsampling thin targets after the input image resize.
        boundary_options = {"fliplr": 0.0, "flipud": 0.0, "copy_paste": 0.0, "mask_ratio": 1}
    results = network.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, workers=workers, project=str(project.resolve()), name=name,
        # Keep the full track mask beneath overlapping car masks.
        overlap_mask=False,
        **boundary_options,
    )
    if upload:
        from .publication import upload_checkpoint

        checkpoint = network.trainer.best
        if not checkpoint.is_file():
            checkpoint = network.trainer.last
        print(f"Uploading trained checkpoint: {checkpoint}", flush=True)
        try:
            record = asyncio.run(upload_checkpoint(checkpoint, name=upload_name))
        except (OSError, ValueError, ImportError, httpx.HTTPError) as exc:
            raise RuntimeError(
                f"Training completed, but backend upload failed: {exc}. "
                f"The local checkpoint is preserved at {checkpoint}. "
                "Use the upload command to publish it without retraining."
            ) from exc
        print(f"Model uploaded to backend: {record['_id']}", flush=True)
    return results
