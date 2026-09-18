"""Local image segmentation training; checkpoints stay in the training workspace."""

from __future__ import annotations

from pathlib import Path

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
    project: Path = WORKSPACE_DIR / "models/image_segmentation",
    name: str = "train",
):
    data = data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    from ultralytics import YOLO

    network = YOLO(model, task="segment")
    if network.task != "segment":
        raise ValueError("Use a segmentation checkpoint or model YAML, such as yolo11n-seg.pt.")
    return network.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, workers=workers, project=str(project.resolve()), name=name,
    )
